import datasets
import transformers.data.metrics.squad_metrics as squad_metrics


def process_results(doc, results):
    preds = results[0]
    reference = doc["answers"]["text"][0]
    f1_sum = squad_metrics.compute_f1(reference, preds)
    exact_match = squad_metrics.compute_exact(reference, preds)
    return {"f1": f1_sum, "exact_match": exact_match}


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["title"] = doc["context"].strip().split("\n")[0].strip()
        doc["passage"] = "\n".join(doc["context"].strip().split("\n")[1:]).strip()
        doc["question"] = " ".join(doc["question"].strip().split())
        return doc

    return dataset.map(_helper)


def unwrap(doc):
    return doc["title"], doc["passage"], doc["question"]


def p0(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f"Tittel: {title}\n\nTekst: {passage}\n\nSpørsmål: {question}\n\nSvar:"
    return prompt


def p1(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f'Tittel: {title}\n\nTekst: {passage}\n\nGitt teksten over, hva er svaret på følgende spørsmål? "{question}"\n\nSvar:'
    return prompt


def p2(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = (
        f"Tittel: {title}\n\nTekst: {passage}\n\nSvar på følgende: {question}\n\nSvar:"
    )
    return prompt


def p3(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f'Tittel: {title}\n\nTekst: {passage}\n\nHvordan kan man svare på spørsmålet "{question}", gitt teksten over?\n\nSvar:'
    return prompt


def p4(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f'Tittel: {title}\n\nTekst:{passage}\n\nGitt teksten over, besvar følgende spørsmål: "{question}"\n\nSvar:'
    return prompt


# p5: instruction
def p5(doc):
    return 'Kontekst: "{context}"\n{question}'.format(**doc)


# p6: instruction
def p6(doc):
    return '{context}\n---\nGitt teksten over, besvar følgende spørsmål: "{question}"'.format(
        **doc
    )


# p7: instruction verbatim
def p7(doc):
    return (
        'Kontekst:"{context}"\n{question}'
        "Svar med kun et ordrett (verbatim) utdrag av konteksten som inneholder svaret."
    ).format(**doc)


def p8(doc):
    return (
        "Answer the question in Norwegian Bokmål with the verbatim excerpt of the context that sufficiently answers the question."
        'Context:"{context}"\n{question}'
    ).format(**doc)
