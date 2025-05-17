from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import string

def normalize(text: str) -> str:
    """
    Lowercase, remove punctuation, and strip whitespace.
    """
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).strip()


def tokenize(text: str) -> list:
    return normalize(text).split()


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    if rouge_scorer is None:
        raise ImportError("rouge_score package not installed")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(gold, pred)
    return scores['rougeL'].fmeasure


def bleu(pred: str, gold: str) -> float:
    if sentence_bleu is None:
        raise ImportError("nltk package not installed or missing BLEU support")
    ref = [tokenize(gold)]
    hyp = tokenize(pred)
    return sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)


def meteor(pred: str, gold: str) -> float:
    if meteor_score is None:
        raise ImportError("nltk package not installed or missing METEOR support")
    return meteor_score([normalize(gold)], normalize(pred))


METRIC_FUNCS = {
    'em': exact_match,
    'f1': f1_score,
    'rouge': rouge_l,
    'bleu': bleu,
    'meteor': meteor,
}
