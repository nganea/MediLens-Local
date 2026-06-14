GENERAL_SAFETY_WARNING = {
    "English": (
        "Do not take this medicine unless it was prescribed for you. "
        "If you are unsure what this medicine is or whether it is safe for you, "
        "ask a pharmacist, doctor, or other qualified healthcare professional."
    ),
    "French": (
        "Ne prenez pas ce medicament sauf s'il vous a ete prescrit. "
        "Si vous n'etes pas sur de ce qu'est ce medicament ou s'il est sans danger pour vous, "
        "demandez conseil a un pharmacien, un medecin ou un autre professionnel de sante qualifie."
    ),
    "German": (
        "Nehmen Sie dieses Arzneimittel nicht ein, es sei denn, es wurde Ihnen verschrieben. "
        "Wenn Sie unsicher sind, was dieses Arzneimittel ist oder ob es fuer Sie sicher ist, "
        "fragen Sie einen Apotheker, Arzt oder eine andere qualifizierte medizinische Fachperson."
    ),
    "Italian": (
        "Non prendere questo medicinale a meno che non sia stato prescritto per te. "
        "Se non sei sicuro di che medicinale sia o se sia sicuro per te, "
        "chiedi a un farmacista, medico o altro professionista sanitario qualificato."
    ),
    "Spanish": (
        "No tome este medicamento a menos que se lo hayan recetado. "
        "Si no esta seguro de que medicamento es o si es seguro para usted, "
        "consulte a un farmaceutico, medico u otro profesional sanitario cualificado."
    ),
    "Romanian": (
        "Nu lua acest medicament decat daca ti-a fost prescris. "
        "Daca nu esti sigur ce este acest medicament sau daca este sigur pentru tine, "
        "intreaba un farmacist, un medic sau un alt profesionist medical calificat."
    ),
}

LOW_CONFIDENCE_MESSAGE = {
    "English": "I could not confidently identify the medicine. Please try a clearer photo or ask a pharmacist.",
    "French": "Je n'ai pas pu identifier ce medicament avec suffisamment de confiance. Essayez une photo plus claire ou demandez conseil a un pharmacien.",
    "German": "Ich konnte das Arzneimittel nicht sicher identifizieren. Bitte versuchen Sie ein klareres Foto oder fragen Sie einen Apotheker.",
    "Italian": "Non sono riuscito a identificare il medicinale con sicurezza. Prova con una foto piu chiara o chiedi a un farmacista.",
    "Spanish": "No pude identificar el medicamento con suficiente confianza. Pruebe con una foto mas clara o consulte a un farmaceutico.",
    "Romanian": "Nu am putut identifica medicamentul cu suficienta incredere. Incearca o fotografie mai clara sau intreaba un farmacist.",
}

CONFIRM_MESSAGE = {
    "English": "Always confirm with a pharmacist, doctor, or the medicine leaflet.",
    "French": "Confirmez toujours avec un pharmacien, un medecin ou la notice du medicament.",
    "German": "Bestaetigen Sie dies immer mit einem Apotheker, Arzt oder der Packungsbeilage.",
    "Italian": "Conferma sempre con un farmacista, un medico o il foglio illustrativo.",
    "Spanish": "Confirme siempre con un farmaceutico, medico o el prospecto del medicamento.",
    "Romanian": "Confirma intotdeauna cu un farmacist, un medic sau prospectul medicamentului.",
}


def no_database_info_message(language: str) -> str:
    messages = {
        "English": "No information is available for this medicine in the local database.",
        "French": "Aucune information n'est disponible pour ce medicament dans la base de donnees locale.",
        "German": "Zu diesem Arzneimittel sind in der lokalen Datenbank keine Informationen verfuegbar.",
        "Italian": "Non sono disponibili informazioni su questo medicinale nel database locale.",
        "Romanian": "Nu exista informatii despre acest medicament in baza de date locala.",
        "Spanish": "No hay informacion disponible sobre este medicamento en la base de datos local.",
    }
    return messages[language]


def manual_entry_prompt(language: str) -> str:
    messages = {
        "English": "N/A\nThe image could not be processed. Please enter the name of the medicine and search again.",
        "French": "N/A\nL'image n'a pas pu etre traitee. Veuillez saisir le nom du medicament et relancer la recherche.",
        "German": "N/A\nDas Bild konnte nicht verarbeitet werden. Bitte geben Sie den Namen des Arzneimittels ein und suchen Sie erneut.",
        "Italian": "N/A\nNon e stato possibile elaborare l'immagine. Inserisci il nome del medicinale e cerca di nuovo.",
        "Romanian": "N/A\nImaginea nu a putut fi procesata. Introdu numele medicamentului si cauta din nou.",
        "Spanish": "N/A\nNo se pudo procesar la imagen. Introduzca el nombre del medicamento y busque de nuevo.",
    }
    return messages[language]

