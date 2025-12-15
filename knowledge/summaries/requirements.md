# Requirements

- Operate via PR-only workflows; no autonomous live trading deployments.
- Maintain deterministic pipelines with Docker-based tooling and recorded hashes.
- Keep secrets out of the repository; use `.env` files and documented secret injection.
- Ensure all experiments log parameters, seeds, dataset hashes, and environment details.
