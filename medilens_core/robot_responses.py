from .pipeline import MedicineIdentification


START_PROMPT = "Okay, hold the medicine label in front of me."
PICTURE_TAKEN = "I have taken a picture. I am checking it now."
NOT_FOUND = "I do not know what this medicine is. Try the MediLens app on your device."


def medicine_found_response(result: MedicineIdentification) -> str:
    return (
        f"It looks like {result.medicine_name}. "
        f"{result.common_uses} "
        "Speak with a pharmacist or doctor if you are not sure it is safe for you."
    )


def medicine_response(result: MedicineIdentification) -> str:
    if not result.found:
        return NOT_FOUND
    return medicine_found_response(result)

