# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Report security vulnerabilities to security@aumos.ai.

Do NOT open public GitHub issues for security vulnerabilities.

## Security Design

- **Tenant Isolation**: Row-level security on all `ctx_*` tables via PostgreSQL RLS
- **Graph Isolation**: Per-tenant AGE graphs (`aumos_ctx_{tenant_id}`) prevent cross-tenant data access
- **Auth**: Bearer JWT validated by aumos-common middleware on every request
- **Embedding API Keys**: Stored in environment variables, never logged or returned in API responses
- **Cypher Query Safety**: Queries are scoped to tenant graph — cross-tenant graph access is architecturally impossible
- **Non-root Container**: Service runs as `aumos` user with minimal permissions
