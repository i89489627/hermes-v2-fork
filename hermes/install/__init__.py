"""hermes-v2 install/update tooling (plan P21.T11).

Holds the update-path logic for the v2 package: config migrations that
preserve user data across releases (issue #25272), backups before any config
change, and callback-based schema-change prompts.
"""
