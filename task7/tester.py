import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import requests
import importlib.util

# Настройка логирования для тестирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RAGTester")

# Путь к rag_engine.py (относительно корня проекта)
rag_engine_path = Path(__file__).parent.parent / "task4" / "rag_engine.py"

# Динамическая загрузка модуля rag_engine
spec = importlib.util.spec_from_file_location("rag_engine", rag_engine_path)
rag_engine_module = importlib.util.module_from_spec(spec)
sys.modules["rag_engine"] = rag_engine_module
spec.loader.exec_module(rag_engine_module)

# Изменяем пути к файлам внутри rag_engine
rag_engine_module.PromptsDir = Path(__file__).parent.parent / "task4" / "prompts"


# Переопределяем методы загрузки промптов, чтобы они использовали правильный путь
def _load_system_prompt_fixed(self) -> str:
    prompts_dir = rag_engine_module.PromptsDir
    return (prompts_dir / "system_prompt.txt").read_text(encoding="utf-8").strip()

def _load_few_shot_examples_fixed(self) -> str:
    prompts_dir = rag_engine_module.PromptsDir
    return (prompts_dir / "few_shot_examples.txt").read_text(encoding="utf-8").strip()


# Заменяем методы в классе
rag_engine_module.RAGEngine._load_system_prompt = _load_system_prompt_fixed
rag_engine_module.RAGEngine._load_few_shot_examples = _load_few_shot_examples_fixed

# Импортируем нужный класс
RAGEngine = rag_engine_module.RAGEngine


class RAGTester:
    def __init__(self, golden_set_path: str, evaluation_prompt_path: str = None):
        self.golden_set_path = Path(golden_set_path)
        self.evaluation_prompt_path = Path(evaluation_prompt_path) if evaluation_prompt_path else Path(__file__).parent / "evaluation_prompt.txt"
        self.engine = self._initialize_engine()
        self.test_results = []
        
    def _initialize_engine(self) -> RAGEngine:

        rag_engine_module.CHROMA_PATH = str(Path(__file__).parent.parent / "task3/chroma_db")
        
        logger.info("🚀 Инициализация RAG-движка...")
        engine = RAGEngine()
        logger.info("✅ RAG-движок успешно инициализирован")
        return engine
    
    def load_golden_set(self) -> list:
        """Загрузка golden set из файла."""
        logger.info(f"📥 Загрузка golden set из {self.golden_set_path}")
        with open(self.golden_set_path, 'r', encoding='utf-8') as f:
            golden_set = json.load(f)
        logger.info(f"✅ Загружено {len(golden_set)} тестовых примеров")
        return golden_set
    
    def load_evaluation_prompt(self) -> str:
        """Загрузка системного промпта для оценки из файла."""
        logger.info(f"📥 Загрузка промпта для оценки из {self.evaluation_prompt_path}")
        with open(self.evaluation_prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        logger.info("✅ Промпт для оценки загружен")
        return prompt
    
    def call_ollama_for_evaluation(self, evaluation_prompt: str) -> dict:

        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "qwen3:4b-instruct",
            "messages": [
                {"role": "system", "content": evaluation_prompt}
            ],
            "stream": False,
            "options": {
                "num_predict": 2048,
                "temperature": 0.3,  # Низкая температура для более детерминированного ответа
                "top_p": 0.8
            }
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result["message"]["content"].strip()
        else:
            raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
    
    def evaluate_response(self, question: str, expected_answer: str, actual_response: str) -> dict:

        # Извлекаем только часть после "Ответ:" из actual_response
        parts = actual_response.split("\n\n")
        answer_part = actual_response
        
        for part in parts:
            if part.strip().startswith("Ответ:") or part.strip().startswith("Answer:"):
                answer_part = part.strip()
                break
        
        # Извлекаем текст после "Ответ:" или "Answer:"
        import re
        match = re.search(r'(?:Ответ|Answer):\s*(.*)', answer_part, re.DOTALL | re.IGNORECASE)
        if match:
            answer_part = match.group(1).strip()
        else:
            # Если не найдено "Ответ:", используем весь текст
            answer_part = actual_response
        
        # Загружаем системный промпт для оценки
        base_prompt = self.load_evaluation_prompt()
        
        # Формируем полный промпт с конкретными данными
        full_prompt = base_prompt.format(
            question=question,
            expected_answer=expected_answer,
            actual_response=answer_part
        )
        
        try:
            evaluation_result = self.call_ollama_for_evaluation(full_prompt)
            # Парсим JSON из ответа
            import ast
            try:
                result_dict = ast.literal_eval(evaluation_result)
            except:
                # Если не получилось распарсить как JSON, пробуем найти JSON в тексте
                import re
                json_match = re.search(r'\{.*\}', evaluation_result, re.DOTALL)
                if json_match:
                    import json as json_module
                    result_dict = json_module.loads(json_match.group())
                else:
                    # Если не нашли JSON, делаем ручную оценку
                    result_dict = {
                        "answer_found": "да" if "информация не найдена" not in answer_part.lower() and len(answer_part.strip()) > 10 else "нет",
                        "completeness_score": 5,  # по умолчанию средняя оценка
                        "meaning_match": "нет",  # по умолчанию
                        "explanation": "Не удалось распарсить оценку от LLM"
                    }
            
            return result_dict
        except Exception as e:
            logger.error(f"Ошибка при оценке ответа: {e}")
            return {
                "answer_found": "нет",
                "completeness_score": 0,
                "meaning_match": "нет",
                "explanation": f"Ошибка при оценке: {str(e)}"
            }
    
    def run_test_suite(self, log_file_path: str = None):

        if log_file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = f"logs/test_results_{timestamp}.log"
        
        # Создаем директорию logs, если она не существует
        log_dir = Path(log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка логгера для результатов тестирования
        test_logger = logging.getLogger("TestResults")
        test_logger.setLevel(logging.INFO)
        
        # Удаляем старые хендлеры
        for handler in test_logger.handlers[:]:
            test_logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(message)s')  # Формат JSON для автоматического анализа
        file_handler.setFormatter(formatter)
        test_logger.addHandler(file_handler)
        test_logger.propagate = False
        
        golden_set = self.load_golden_set()
        
        total_tests = len(golden_set)
        successful_answers = 0
        meaning_matches = 0
        
        # Записываем информацию о начале тестирования в JSON
        start_info = {
            "event_type": "test_suite_start",
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "golden_set_path": str(self.golden_set_path),
            "evaluation_prompt_path": str(self.evaluation_prompt_path)
        }
        test_logger.info(json.dumps(start_info, ensure_ascii=False))
        
        for idx, test_case in enumerate(golden_set, 1):
            question = test_case["question"]
            expected_answer = test_case["expected_answer"]
            
            try:
                # Получаем ответ от RAG-системы
                actual_response = self.engine.query(question)
                
                # Оцениваем ответ
                evaluation = self.evaluate_response(question, expected_answer, actual_response)
                
                # Считаем статистику
                if evaluation['answer_found'] == 'да':
                    successful_answers += 1
                if evaluation['meaning_match'] == 'да':
                    meaning_matches += 1
                
                # Сохраняем результат теста
                test_result = {
                    "event_type": "test_result",
                    "test_number": idx,
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_response": actual_response,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat()
                }
                self.test_results.append(test_result)
                
                # Записываем результат в лог
                test_logger.info(json.dumps(test_result, ensure_ascii=False))
                
            except Exception as e:
                test_result = {
                    "event_type": "test_error",
                    "test_number": idx,
                    "question": question,
                    "expected_answer": expected_answer,
                    "actual_response": f"Ошибка: {str(e)}",
                    "evaluation": {
                        "answer_found": "нет",
                        "completeness_score": 0,
                        "meaning_match": "нет",
                        "explanation": f"Ошибка выполнения: {str(e)}"
                    },
                    "timestamp": datetime.now().isoformat()
                }
                test_logger.info(json.dumps(test_result, ensure_ascii=False))
                self.test_results.append(test_result)
        
        # Выводим итоговую статистику
        end_info = {
            "event_type": "test_suite_end",
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful_answers": successful_answers,
            "meaning_matches": meaning_matches,
            "success_rate": successful_answers / total_tests if total_tests > 0 else 0,
            "meaning_match_rate": meaning_matches / total_tests if total_tests > 0 else 0,
            "average_completeness_score": sum([r['evaluation']['completeness_score'] for r in self.test_results if 'evaluation' in r and isinstance(r['evaluation'], dict)]) / len([r for r in self.test_results if 'evaluation' in r and isinstance(r['evaluation'], dict)]) if len([r for r in self.test_results if 'evaluation' in r and isinstance(r['evaluation'], dict)]) > 0 else 0
        }
        test_logger.info(json.dumps(end_info, ensure_ascii=False))
        
        logger.info(f"✅ Тестирование завершено. Результаты сохранены в {log_file_path}")
        
        return self.test_results


if __name__ == "__main__":
    # Путь к golden_set.json
    golden_set_path = Path(__file__).parent / "golden_set.json"
    
    # Путь к файлу с системным промптом для оценки
    evaluation_prompt_path = Path(__file__).parent / "evaluation_prompt.txt"
    
    if not golden_set_path.exists():
        logger.error(f"❌ Файл {golden_set_path} не найден!")
        sys.exit(1)
    
    if not evaluation_prompt_path.exists():
        logger.error(f"❌ Файл {evaluation_prompt_path} не найден!")
        sys.exit(1)
    
    # Создаем тестировщик
    tester = RAGTester(
        golden_set_path=str(golden_set_path),
        evaluation_prompt_path=str(evaluation_prompt_path)
    )
    
    # Запускаем тестирование
    logger.info("🚀 Начало автоматического тестирования RAG-системы")
    results = tester.run_test_suite()
    
    logger.info("✅ Автоматическое тестирование завершено")