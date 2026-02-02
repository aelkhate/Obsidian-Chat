# src/vault_policy.py

VAULT_ONTOLOGY = {
    "para": {
        "projects_root": "03_projects",
        "areas_root": "02_areas",
        "resources_root": "04_resources",
        "archive_root": "05_archive",
    },
    # interpretation rules
    "semantics": {
        "active_projects_definition": "Projects are considered active if they are under 03_projects. Inactive/completed projects are moved to 05_archive.",
        "archive_is_inactive": True,
    }
}
