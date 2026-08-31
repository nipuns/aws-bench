"""Shared models for the CCAPI package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from botocore.config import Config

# Concurrency limits for thread pools.
# HEAVY: paginated CCAPI list_resources calls — lower to avoid throttling.
# LIGHT: fast or fail-fast operations (filtering, deletion, type discovery).
# ACCOUNT: outer-loop concurrency for multi-account operations that internally
#          fan out per-region. Each account is a distinct AWS account, so its
#          calls hit distinct throttle buckets — this bound is about thread cost
#          (it nests over the per-region HEAVY pool), not throttling.
MAX_WORKERS_HEAVY = 4
MAX_WORKERS_LIGHT = 10
MAX_WORKERS_ACCOUNT = 4

CCAPI_CLIENT_CONFIG = Config(
    retries={"max_attempts": 8, "mode": "adaptive"},
    max_pool_connections=max(MAX_WORKERS_HEAVY, MAX_WORKERS_LIGHT) + 10,
)

# Client config for the fail-closed GetResource existence check ONLY. Retries are
# disabled here (``max_attempts: 0`` == one attempt, no retries): an existence check
# resolves any non-definitive outcome — throttle, unsupported type, transient error,
# or a server-side handler InternalFailure — by KEEPING the resource, so botocore
# retrying can only ever turn a "keep" into a possible "drop" of a phantom. The
# broken-handler types (AWS::ControlTower::EnabledBaseline, AWS::SecurityHub::Standard)
# return HandlerInternalFailureException, modeled as a 5xx server fault, which the
# adaptive retry mode retries up to 8 times (~18s/check) on every cleanup sweep
# wave — pure waste, since the handler fails identically every time. The scan/list
# path keeps CCAPI_CLIENT_CONFIG's retries (bulk listing genuinely benefits from
# throttle backoff); this narrower config applies only to the per-resource check.
EXISTENCE_CHECK_CLIENT_CONFIG = Config(
    retries={"max_attempts": 0},
    max_pool_connections=max(MAX_WORKERS_HEAVY, MAX_WORKERS_LIGHT) + 10,
)

CUSTOM_RESOURCE_PREFIX = "Custom::"
SERVICE_ROLE_PREFIX = "AWSServiceRole"

# Roles that must never be deleted, even if absent from the baseline snapshot.
# OrganizationAccountAccessRole is the role the management account uses to
# assume into member accounts; deleting it permanently severs control of the
# member account.
PROTECTED_IAM_ROLE_NAMES = frozenset({"OrganizationAccountAccessRole"})

# Account-global (non-regional) resource types. Their identifiers are identical
# across every region, so a per-region scan surfaces the same resource in every
# region. On reset each region would otherwise race to delete it — and a global
# resource a regional resource depends on (e.g. a Bedrock knowledge base's IAM
# execution role) can be deleted by a region that has no such dependent while
# another region is still tearing the dependent down, wedging that delete. These
# types are therefore deleted once, in a final account-level pass, after every
# region's resource deletion has completed. See reset.manager and cleanup.manager
# (the cleanup path withholds them from its concurrent per-region sweeps and reaps
# them once behind a barrier via ``_sweep_global_leftovers``).
GLOBAL_RESOURCE_TYPES = frozenset(
    {
        "AWS::IAM::Role",
        "AWS::IAM::ManagedPolicy",
        "AWS::IAM::User",
        "AWS::IAM::Group",
        "AWS::IAM::InstanceProfile",
        "AWS::IAM::OIDCProvider",
        "AWS::IAM::SAMLProvider",
        "AWS::IAM::ServerCertificate",
        "AWS::IAM::VirtualMFADevice",
        "AWS::ResourceExplorer2::Index",
        "AWS::Route53::HostedZone",
        "AWS::CloudFront::Distribution",
    }
)
LOG_TRUNCATE_SHORT = 50
LOG_TRUNCATE_MEDIUM = 60
LOG_TRUNCATE_LONG = 80

# CCAPI error codes that indicate a resource type is permanently unsupported.
UNSUPPORTED_CCAPI_ERROR_CODES = frozenset(
    {
        "UnsupportedActionException",
        "TypeNotFoundException",
        "ValidationException",
        "InvalidRequestException",
    }
)

# Transient throttle codes: rate-limited, not rejected. Kept distinct from UNSUPPORTED_* so an
# existence check maps them to UNKNOWN (unverified), never SKIPPED.
THROTTLE_ERROR_CODES = frozenset(
    {
        "ThrottlingException",
        "Throttling",
        "TooManyRequestsException",
        "RequestLimitExceeded",
    }
)

# Server-side handler faults: Cloud Control invoked the resource type's handler and it
# failed internally (HandlerErrorCode InternalFailure). Modeled as a 5xx fault, so the
# adaptive retry mode retries it to exhaustion. A fixed set of default resource types
# (AWS::ControlTower::EnabledBaseline, AWS::SecurityHub::Standard, ...) have handlers
# broken server-side and fail this way on EVERY GetResource. Kept distinct from
# UNSUPPORTED_* so an existence check maps them to UNKNOWN (keep the resource), never
# SKIPPED — the type is not unsupported, its handler is merely broken, so a real orphan
# of the type must still be attempted rather than silently leaked.
HANDLER_FAILURE_ERROR_CODES = frozenset(
    {
        "HandlerInternalFailureException",
        "InternalFailure",
    }
)


@dataclass(frozen=True)
class Resource:
    """A single AWS resource identified by type and identifier."""

    type: str
    identifier: str


@dataclass(frozen=True)
class DeletionFailureEvent:
    """Represents a deletion failure."""

    status_message: str

    @classmethod
    def from_ccapi_event(cls, event: dict[str, Any]) -> DeletionFailureEvent:
        """Create from a CCAPI ProgressEvent dict."""
        return cls(status_message=event.get("StatusMessage", "Unknown error"))


@dataclass
class ScanResult:
    """Result of scanning an account for resources via CCAPI."""

    detected: dict[str, list[dict]]
    failed: dict[str, str]
    empty: set[str] = field(
        default_factory=set
    )  # Resource types that were scanned but returned 0 resources


@dataclass
class SubmitResult:
    """Result of submitting deletion requests to CCAPI."""

    tokens: dict[str, Resource]
    failures: dict[Resource, DeletionFailureEvent]
    # Resources a concurrent op is already deleting — dropped from retries, not failures.
    already_handled: set[Resource] = field(default_factory=set)


@dataclass
class PollResult:
    """Result of polling CCAPI deletion requests."""

    succeeded: set[str]
    pending: set[str]
    failed: dict[str, DeletionFailureEvent]
