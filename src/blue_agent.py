# blue_agent.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ==========================================================
# BLUE AGENT
# SEQUENCE CLASSIFICATION VERSION
# ==========================================================

class BlueAgent:

    def __init__(
        self,
        model_type="deberta",
        model_dir=None,
        max_length=2048,
        threshold=0.35,
        device="auto",
        load_in_4bit=True,
        batch_size=32,
    ):

        self.model_type = model_type.lower()
        self.max_length = max_length
        self.threshold = threshold
        self.load_in_4bit = load_in_4bit
        self.batch_size = batch_size

        # ======================================================
        # DEVICE
        # ======================================================

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🔧 BlueAgent using device: {self.device.upper()}")

        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"   Memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

        # ======================================================
        # HISTORY / CACHE
        # ======================================================

        self.risk_history = []
        self.level_history = []
        self.cache = {}

        # ======================================================
        # MODEL CONFIG
        # ======================================================

        #
        # IMPORTANT:
        # use sequence classification model
        #

        if self.model_type == "deberta":

            #
            # RECOMMENDED:
            # microsoft/deberta-v3-base + your finetune
            #
            # TEMP:
            # use sentiment model for demo
            #

            self.base_model_path = (
                "cardiffnlp/twitter-roberta-base-offensive"
            )

            self.peft_model_path = None

        elif self.model_type == "qwen":

            if model_dir is None:
                raise ValueError(
                    "model_dir is required for qwen model"
                )

            self.base_model_path = os.path.join(
                model_dir,
                "models",
                "Qwen3-4B-Instruct-2507"
            )

            self.peft_model_path = os.path.join(
                model_dir,
                "lora_adapter"
            )

        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}"
            )

        self._load_model()

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def _load_model(self):

        print(f"📥 Loading tokenizer from {self.base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
        }

        # ======================================================
        # QUANTIZATION
        # ======================================================

        if self.load_in_4bit and self.device == "cuda":

            print("🔧 Using 4-bit quantization")

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            model_kwargs["device_map"] = "auto"

        print(f"📥 Loading model from {self.base_model_path}")

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_path,
            **model_kwargs
        )

        #
        # OPTIONAL PEFT
        #

        if self.model_type == "qwen" and self.peft_model_path:

            print("📥 Loading PEFT adapter")

            self.model = PeftModel.from_pretrained(
                base_model,
                self.peft_model_path
            )

        else:
            self.model = base_model

        #
        # MOVE TO DEVICE
        #

        if not (
            self.load_in_4bit and self.device == "cuda"
        ):
            self.model = self.model.to(self.device)

        self.model.eval()

        print("✅ Model loaded successfully")

    # ======================================================
    # LABELS
    # ======================================================

    def _risk_label(self, score: float) -> str:

        if score < 0.2:
            return "SAFE"

        elif score < 0.45:
            return "LOW"

        elif score < 0.7:
            return "MEDIUM"

        return "HIGH"

    # ======================================================
    # HEURISTICS
    # ======================================================

    def _heuristic_boost(self, text: str) -> float:

        text = text.lower()

        dangerous_patterns = [
            # Русские угрозы и насилие
            "срочно", "последний шанс", "угроза", "убью", "убить", "взрыв", "оружие", "наркотики",
            "переведи деньги", "скинь карту", "обман", "мошенничество", "ненависть", "экстремизм",
            "насилие", "атака", "террор", "смерть", "зарезать", "взорвать", "поджечь", "похитить",
            "шантаж", "вымогательство", "напасть", "расправа", "ликвидировать", "труп",

            # Финансовые манипуляции
            "лотерея", "выигрыш", "приз", "деньги", "кошелек", "биткоин", "криптовалюта",
            "инвестиции", "заработок", "пассивный доход", "легкие деньги", "быстрый заработок",
            "финансовая пирамида", "лохотрон", "развод", "обналичить",

            # Манипулятивные фразы
            "только сегодня", "успей", "ограниченное предложение", "секрет", "раскрой правду",
            "власть", "контроль", "запугивание", "шок", "сенсация", "правда которую скрывают",
            "они не хотят чтобы ты знал", "система лжёт", "доказательство", "разоблачение",

            # Оскорбления и враждебность
            "идиот", "дебил", "тупой", "лох", "предатель", "фашист", "нацист", "расист",
            "гомофоб", "ксенофоб", "урод", "тварь", "сволочь", "подонок",

            # Призывы к действию
            "блокируй", "распространи", "обязательно", "требую", "заставь", "объяви бойкот",
            "выступи против", "собери подписи", "жми на репост", "не молчи",'ВНИМАНИЕ', 'ТРЕВОГА', ''

            # Английские аналоги (оставляем для смешанных текстов)
            "danger", "urgent", "kill", "weapon", "fraud", "scam", "hate", "attack", "terror",
            "death", "explosion", "money", "bitcoin", "crypto", "investment", "lottery", "winner",
            "secret", "truth", "exposed", "boycott", "block", "spread"
        ]

        boost = 0.0

        for pattern in dangerous_patterns:

            if pattern in text:
                boost += 0.15

        return min(boost, 0.5)

    # ======================================================
    # ANALYZE BATCH
    # ======================================================

    def analyze_batch(self, texts):

        final_results = [None] * len(texts)

        uncached_texts = []
        uncached_indices = []

        #
        # CACHE
        #

        for idx, text in enumerate(texts):

            if text in self.cache:
                final_results[idx] = self.cache[text]

            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)

        if not uncached_texts:
            return final_results

        #
        # TOKENIZE
        #

        enc = self.tokenizer(
            uncached_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        enc = {
            k: v.to(self.device)
            for k, v in enc.items()
        }

        #
        # INFERENCE
        #

        with torch.no_grad():

            outputs = self.model(**enc)

            logits = outputs.logits

            #
            # binary
            #

            if logits.shape[-1] == 1:

                probs = torch.sigmoid(
                    logits.squeeze(-1)
                )

            else:

                probs = torch.softmax(
                    logits,
                    dim=-1
                )[:, -1]

            probs = probs.detach().cpu().numpy()

        #
        # BUILD RESULTS
        #

        for idx, text, score in zip(
            uncached_indices,
            uncached_texts,
            probs
        ):

            score = float(score)

            #
            # heuristic boost
            #

            score += self._heuristic_boost(text)

            score = float(np.clip(score, 0.0, 1.0))

            #
            # suspicious words
            #

            suspicious = []

            for word in re.findall(r"\w+", text):

                word_l = word.lower()

                if any(
                    k in word_l
                    for k in [
                        "угроз",
                        "уб",
                        "оруж",
                        "наркот",
                        "мошен",
                        "scam",
                        "kill",
                        "fraud",
                        "hate",
                    ]
                ):
                    suspicious.append(
                        (word, round(score, 3))
                    )

            result = {
                "RiskScore": round(score, 4),
                "Level": self._risk_label(score),
                "sus_words": suspicious,
            }

            self.cache[text] = result
            final_results[idx] = result

        return final_results

    # ======================================================
    # ANALYZE SINGLE
    # ======================================================

    def analyze(self, messages):

        if isinstance(messages, list):
            text = " ".join(messages)
        else:
            text = messages

        return self.analyze_batch([text])[0]

    # ======================================================
    # STREAM PROCESSING
    # ======================================================

    def process_event(self, text: str):

        result = self.analyze(text)

        self.risk_history.append(
            result["RiskScore"]
        )

        self.level_history.append(
            result["Level"]
        )

        return (
            result["RiskScore"],
            result["Level"]
        )

    # ======================================================
    # PROCESS BATCH
    # ======================================================

    def process_batch(self, texts):

        results = self.analyze_batch(texts)

        for result in results:

            self.risk_history.append(
                result["RiskScore"]
            )

            self.level_history.append(
                result["Level"]
            )

        return [
            (
                r["RiskScore"],
                r["Level"]
            )
            for r in results
        ]

    # ======================================================
    # GLOBAL SUMMARY
    # ======================================================

    def global_summary(self):

        if not self.risk_history:

            return {
                "global_risk_score": 0.0,
                "global_risk_level": "SAFE",
            }

        score = float(
            np.mean(self.risk_history)
        )

        return {
            "global_risk_score": round(score, 4),
            "global_risk_level": self._risk_label(score),
        }

    # ======================================================
    # CACHE
    # ======================================================

    def clear_cache(self):

        self.cache = {}

        print("🗑️ Cache cleared")

    # ======================================================
    # STATS
    # ======================================================

    def get_stats(self):

        return {
            "total_analyzed": len(self.risk_history),
            "cache_size": len(self.cache),
            "avg_risk": (
                np.mean(self.risk_history)
                if self.risk_history
                else 0
            ),
            "device": self.device,
            "model_type": self.model_type,
        }