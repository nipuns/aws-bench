"""Cloud Control API manager — central entry point for all CCAPI operations."""

from __future__ import annotations

import boto3

from aws_bench.logging.logger import get_logger
from aws_bench.resource_management.ccapi.deleter import Deleter
from aws_bench.resource_management.ccapi.exceptions import (
    ResourceExistenceCheckError,
    ResourceExistenceHandlerFailureError,
    ResourceExistenceThrottledError,
    ResourceExistenceUnsupportedError,
    is_not_found_error,
)
from aws_bench.resource_management.ccapi.models import (
    CCAPI_CLIENT_CONFIG,
    EXISTENCE_CHECK_CLIENT_CONFIG,
    HANDLER_FAILURE_ERROR_CODES,
    THROTTLE_ERROR_CODES,
    UNSUPPORTED_CCAPI_ERROR_CODES,
    DeletionFailureEvent,
    Resource,
    ScanResult,
)
from aws_bench.resource_management.ccapi.scanner import Scanner
from aws_bench.resource_management.ccapi.type_registry import TypeRegistry
from aws_bench.utils.concurrent import build_client

logger = get_logger(__name__)


class CloudControlManager:
    """Central manager for AWS Cloud Control API operations."""

    def __init__(self, session: boto3.Session, region_name: str | None = None) -> None:
        """Initialize with a boto3 session and optional region.

        Args:
            session: boto3 Session for AWS operations
            region_name: AWS region name (optional, uses session's region if not provided)
        """
        self._session = session
        self._region_name = region_name or session.region_name
        self._client = build_client(
            session, "cloudcontrol", region_name=self._region_name, config=CCAPI_CLIENT_CONFIG
        )
        # Dedicated client for the fail-closed existence check (resource_exists) with retries
        # disabled — see EXISTENCE_CHECK_CLIENT_CONFIG. The scan/list/delete paths keep the
        # retrying self._client; only the per-resource GetResource check runs no-retry, so a
        # broken-handler type no longer burns 8 adaptive retries per check.
        self._existence_client = build_client(
            session,
            "cloudcontrol",
            region_name=self._region_name,
            config=EXISTENCE_CHECK_CLIENT_CONFIG,
        )
        self._scanner = Scanner(self._client, session=session, region_name=self._region_name)
        self._deleter = Deleter(self._client, resource_exists_fn=self.resource_exists)
        self._type_registry = TypeRegistry(
            session, scan_fn=self._scanner.scan_resources, region_name=self._region_name
        )

    def get_scannable_types(self) -> list[str]:
        """Get all CCAPI-supported resource types, excluding known problematic ones."""
        return self._scanner.get_scannable_types()

    def scan_resources(
        self,
        resource_types: list[str] | None = None,
    ) -> ScanResult:
        """Scan account for resources. If resource_types is None, scans all scannable types."""
        return self._scanner.scan_resources(resource_types)

    def generate_skip_types(self) -> set[str]:
        """Scan all AWS resource types and persist those that should be skipped."""
        return self._type_registry.generate_skip_types()

    def resource_exists(self, resource: Resource) -> bool:
        """Check if a resource exists via CCAPI get_resource.

        Uses the no-retry existence client (``EXISTENCE_CHECK_CLIENT_CONFIG``): this is a
        fail-closed verification whose caller keeps the resource on any non-definitive
        outcome, so retrying a doomed check (notably a broken-handler InternalFailure) only
        burns latency without changing the decision.
        """
        try:
            self._existence_client.get_resource(
                TypeName=resource.type, Identifier=resource.identifier
            )
            return True
        except self._existence_client.exceptions.ResourceNotFoundException:
            return False
        except Exception as exc:
            error_code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if error_code in THROTTLE_ERROR_CODES:
                logger.debug("CCAPI throttled checking %s: %s", resource.type, error_code)
                raise ResourceExistenceThrottledError(
                    f"CCAPI throttled checking {resource.type} '{resource.identifier}': "
                    f"{error_code}"
                ) from exc
            if error_code in UNSUPPORTED_CCAPI_ERROR_CODES:
                # Expected for every CCAPI-unsupported type, high-volume — TRACE:
                # kept in the ledger run.log, filtered off the DEBUG job/trial sinks.
                logger.trace("CCAPI does not support %s: %s", resource.type, error_code)
                raise ResourceExistenceUnsupportedError(
                    f"CCAPI does not support {resource.type}: {error_code}"
                ) from exc
            if error_code in HANDLER_FAILURE_ERROR_CODES:
                # A type whose CCAPI handler is broken server-side (fails every check).
                # High-volume during cleanup verification — TRACE, like the unsupported case.
                logger.trace("CCAPI handler failed for %s: %s", resource.type, error_code)
                raise ResourceExistenceHandlerFailureError(
                    f"CCAPI handler failed for {resource.type} '{resource.identifier}': "
                    f"{error_code}"
                ) from exc
            if is_not_found_error(exc):
                logger.debug("Resource gone: %s '%s'", resource.type, resource.identifier)
                return False
            logger.debug(
                "Failed to check existence of %s '%s': %s",
                resource.type,
                resource.identifier,
                exc,
            )
            raise ResourceExistenceCheckError(
                f"Failed to check existence of {resource.type} '{resource.identifier}': {exc}"
            ) from exc

    def delete_resources(self, resources: list[Resource]) -> dict[Resource, DeletionFailureEvent]:
        """Delete resources via CCAPI with ordered batching and retry.

        Returns dict of Resource → failure for resources that couldn't be deleted.
        """
        return self._deleter.delete_resources(resources)
