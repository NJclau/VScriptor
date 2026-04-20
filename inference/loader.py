"""Model loader helpers."""


def get_model_and_processor() -> tuple[None, None]:
    """Return model and processor singletons.

    This repository currently runs a lightweight placeholder pipeline and does not
    require heavy model objects for orchestration tests.
    """
    return None, None
