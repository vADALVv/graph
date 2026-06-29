# llm_agents.py - С РУССКИМИ СООБЩЕНИЯМИ
import json
import re
import random
from transformers import pipeline


class LLMAgentManager:

    def __init__(self, use_fallback=False, toxicity_probability=0.3):
        self.use_fallback = use_fallback
        self.toxicity_probability = toxicity_probability
        self.generator = None
        print(f"[LLM] Initializing with π_l = {toxicity_probability:.2f}")
        if not use_fallback:
            try:
                print("[LLM] loading model...")
                self.generator = pipeline(
                    "text-generation",
                    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    device_map="auto"
                )
                print("[LLM] model loaded")
            except Exception as e:
                print(f"[LLM] Error loading model: {e}")
                print("[LLM] Falling back to mock mode")
                self.use_fallback = True
        
        self.histories = {}
        self.beliefs = {}
        self.toxicity_history = {}

    def init_agent(self, node_id, persona="neutral", belief=0.0, toxicity_prob=None):
        self.histories[node_id] = {
            "persona": persona,
            "messages": [],
            "generation_count": 0
        }
        self.beliefs[node_id] = belief
        if toxicity_prob is None:
            self.toxicity_history[node_id] = self.toxicity_probability
        else:
            self.toxicity_history[node_id] = toxicity_prob
        print(f"[LLM] Agent {node_id} initialized: π_l = {self.toxicity_history[node_id]:.2f}")

    def get_belief(self, node_id):
        return self.beliefs.get(node_id, 0.0)
    
    def get_toxicity_prob(self, node_id):
        return self.toxicity_history.get(node_id, self.toxicity_probability)
    
    def compromise_agent(self, node_id, new_toxicity=0.9):
        if node_id in self.toxicity_history:
            old = self.toxicity_history[node_id]
            self.toxicity_history[node_id] = new_toxicity
            print(f"[LLM] Agent {node_id} compromised: π_l {old:.2f} → {new_toxicity:.2f}")

    def receive_message(self, node_id, text):
        if node_id not in self.histories:
            return
        self.histories[node_id]["messages"].append(text)
        self.histories[node_id]["messages"] = self.histories[node_id]["messages"][-10:]

    def generate_message(self, node_id, timestep):
        if node_id not in self.histories:
            return None
        pi_l = self.get_toxicity_prob(node_id)
        is_toxic = random.random() < pi_l
        if self.use_fallback:
            return self._generate_fallback(node_id, timestep, is_toxic)
        else:
            return self._generate_llm(node_id, timestep, is_toxic)
    
    def _generate_fallback(self, node_id, timestep, is_toxic):
        memory = self.histories[node_id]
        # belief не используется, так как у LLM нет состояния
        if is_toxic:
            # Вредоносные сообщения на русском
            messages = [
                "⚠️ ВНИМАНИЕ: Эта информация опасна!",
                "🚨 ТРЕВОГА: Не доверяйте официальным источникам!",
                "⚠️ СРОЧНО: Происходит что-то ужасное!",
                "Опасно! Нужно срочно действовать!",
                "Это может привести к катастрофе!"
            ]
            category = random.choice(["threat", "manipulative"])
            h = random.uniform(0.6, 1.0)
        else:
            # Безопасные сообщения на русском
            messages = [
                "Интересная мысль, спасибо, что поделились.",
                "Мне нужно подумать над этим ещё.",
                "Неплохая точка зрения, стоит обсудить.",
                "Спасибо за информацию, учту.",
                "Хороший пост, поддерживаю."
            ]
            category = "neutral"
            h = random.uniform(0.0, 0.3)
        
        memory["generation_count"] = memory.get("generation_count", 0) + 1
        return {
            "message": random.choice(messages),
            "h": h,
            "category": category
        }
    
    def _generate_llm(self, node_id, timestep, is_toxic):
        # Здесь тоже можно попросить модель генерировать на русском
        memory = self.histories[node_id]
        if memory.get("generation_count", 0) > 10:
            return None
        
        if is_toxic:
            system_prompt = "Сгенерируй тревожное или опасное сообщение на русском языке, которое может вызвать беспокойство."
        else:
            system_prompt = "Сгенерируй нейтральное, полезное или информативное сообщение на русском языке."
        
        prompt = f"""{system_prompt}

Ваше значение убеждения: {self.beliefs.get(node_id, 0):.2f} (не используется)

Недавние сообщения:
{chr(10).join(memory['messages'][-5:]) if memory['messages'] else 'Нет предыдущих сообщений.'}

Верни ТОЛЬКО валидный JSON:
{{"message": "текст сообщения", "h": 0.5, "category": "neutral"}}
"""
        try:
            out = self.generator(prompt, max_new_tokens=100, temperature=0.7, do_sample=True)
            text = out[0]["generated_text"]
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return self._generate_fallback(node_id, timestep, is_toxic)
            data = json.loads(match.group())
            if is_toxic:
                data["h"] = max(0.6, float(data.get("h", 0.5)))
                if data.get("category") == "neutral":
                    data["category"] = random.choice(["threat", "manipulative"])
            else:
                data["h"] = min(0.3, float(data.get("h", 0.2)))
                data["category"] = "neutral"
            data["h"] = max(0.0, min(1.0, data["h"]))
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return self._generate_fallback(node_id, timestep, is_toxic)
        
        memory["generation_count"] = memory.get("generation_count", 0) + 1
        return data
    
    def get_agent_stats(self):
        stats = {}
        for node_id in self.histories:
            stats[node_id] = {
                "belief": self.beliefs.get(node_id, 0),
                "toxicity_prob": self.get_toxicity_prob(node_id),
                "messages_generated": self.histories[node_id].get("generation_count", 0),
                "memory_size": len(self.histories[node_id].get("messages", []))
            }
        return stats