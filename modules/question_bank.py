"""
HireMind AI - Question Bank (Normal Mode)
==========================================
Static question bank with 10 questions:
  - 3 Coding Questions  (always included)
  - 4 Technical Questions (pool of 4)
  - 3 Behavioral Questions (pool of 3)

Each interview session selects 7 questions:
  - All 3 coding questions (mandatory)
  - 4 randomly selected from the remaining 7 (Technical + Behavioral)
"""

import random
from typing import List

# ── CODING QUESTIONS (Always included, all 3) ──────────────────────────────────

CODING_QUESTIONS = [
    {
        "id": "C1",
        "type": "coding",
        "title": "Reverse a String",
        "description": (
            "Write a Python function `reverse_string(s)` that takes a string `s` "
            "as input and returns its reverse.\n\n"
            "**Example:**\n"
            "```\nInput: 'hello'\nOutput: 'olleh'\n```"
        ),
        "starter_code": "def reverse_string(s):\n    # Write your solution here\n    pass",
        "expected_answer": "def reverse_string(s):\n    return s[::-1]",
        "keywords": ["return", "s[::-1]", "reverse", "def reverse_string"],
        "test_cases": [
            {"input": "hello",    "expected": "olleh"},
            {"input": "HireMind", "expected": "dniMeriH"},
            {"input": "",         "expected": ""},
            {"input": "a",        "expected": "a"},
        ],
    },
    {
        "id": "C2",
        "type": "coding",
        "title": "Find Duplicates in a List",
        "description": (
            "Write a Python function `find_duplicates(nums)` that returns a **sorted list** "
            "of all numbers that appear more than once in the input list `nums`.\n\n"
            "**Example:**\n"
            "```\nInput: [1, 2, 3, 2, 4, 3]\nOutput: [2, 3]\n```"
        ),
        "starter_code": "def find_duplicates(nums):\n    # Write your solution here\n    pass",
        "expected_answer": (
            "def find_duplicates(nums):\n"
            "    seen = set()\n"
            "    dupes = set()\n"
            "    for n in nums:\n"
            "        if n in seen:\n"
            "            dupes.add(n)\n"
            "        seen.add(n)\n"
            "    return sorted(dupes)"
        ),
        "keywords": ["return", "set", "def find_duplicates", "for", "append", "sorted"],
        "test_cases": [
            {"input": [1, 2, 3, 2, 4, 3], "expected": [2, 3]},
            {"input": [1, 1, 1],           "expected": [1]},
            {"input": [1, 2, 3],           "expected": []},
        ],
    },
    {
        "id": "C3",
        "type": "coding",
        "title": "Check if a Number is Prime",
        "description": (
            "Write a function `is_prime(n)` that returns `True` if `n` is a prime number, "
            "and `False` otherwise. A prime number is greater than 1 and has no divisors "
            "other than 1 and itself.\n\n"
            "**Example:**\n"
            "```\nInput: 7 → True\nInput: 10 → False\n```"
        ),
        "starter_code": "def is_prime(n):\n    # Write your solution here\n    pass",
        "expected_answer": (
            "def is_prime(n):\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for i in range(2, int(n ** 0.5) + 1):\n"
            "        if n % i == 0:\n"
            "            return False\n"
            "    return True"
        ),
        "keywords": ["return", "def is_prime", "for", "range", "False", "True", "n < 2"],
        "test_cases": [
            {"input": 7,  "expected": True},
            {"input": 1,  "expected": False},
            {"input": 10, "expected": False},
            {"input": 2,  "expected": True},
            {"input": 0,  "expected": False},
        ],
    },
]

# ── TECHNICAL QUESTIONS (Pool of 4) ───────────────────────────────────────────

TECHNICAL_QUESTIONS = [
    {
        "id": "T1",
        "type": "technical",
        "title": "OS Deadlocks",
        "description": (
            "Explain what a **deadlock** is in an Operating System. "
            "What are the **four necessary conditions** (Coffman's conditions) for a deadlock to occur?"
        ),
        "expected_answer": (
            "A deadlock is a situation where two or more processes are blocked, each waiting for a "
            "resource held by another, creating a cycle of dependencies. "
            "The four necessary conditions are: "
            "1. Mutual Exclusion – only one process can use a resource at a time. "
            "2. Hold and Wait – a process holds at least one resource while waiting for others. "
            "3. No Preemption – resources cannot be forcibly taken from a process. "
            "4. Circular Wait – a closed chain of processes exists where each holds a resource needed by the next."
        ),
        "keywords": [
            "mutual exclusion", "hold and wait", "no preemption", "circular wait",
            "deadlock", "blocked", "process", "resource"
        ],
    },
    {
        "id": "T2",
        "type": "technical",
        "title": "SQL vs NoSQL",
        "description": (
            "Compare **SQL** and **NoSQL** databases. "
            "Describe their key differences and state when you would choose each one."
        ),
        "expected_answer": (
            "SQL databases are relational and use a fixed schema with structured tables. They support ACID "
            "transactions and are best for complex queries and consistency-critical applications like banking. "
            "NoSQL databases are schema-less and support flexible data models like documents, graphs, and key-value stores. "
            "They scale horizontally and are suited for large-scale, high-velocity data like social media or real-time analytics."
        ),
        "keywords": [
            "relational", "schema", "acid", "structured", "transactions",
            "nosql", "schemaless", "horizontal scaling", "document", "key-value"
        ],
    },
    {
        "id": "T3",
        "type": "technical",
        "title": "RESTful API Principles",
        "description": (
            "What are the **core architectural principles** of a RESTful API? "
            "List the common HTTP methods used and describe what each one does."
        ),
        "expected_answer": (
            "REST (Representational State Transfer) relies on: Statelessness, Client-Server separation, "
            "Uniform Interface, Cacheability, Layered System, and Code on Demand. "
            "HTTP Methods: GET (retrieve), POST (create), PUT (update/replace), PATCH (partial update), "
            "DELETE (remove). Resources are identified by URIs and representations are usually JSON or XML."
        ),
        "keywords": [
            "stateless", "client-server", "get", "post", "put", "delete", "patch",
            "uniform interface", "uri", "rest", "json", "cacheable"
        ],
    },
    {
        "id": "T4",
        "type": "technical",
        "title": "OOP Concepts",
        "description": (
            "Explain the **four core principles of Object-Oriented Programming (OOP)**. "
            "Give a brief real-world example for each."
        ),
        "expected_answer": (
            "The four pillars of OOP are: "
            "1. Encapsulation – bundling data and methods within a class, hiding internal details. "
            "2. Inheritance – a class inheriting attributes and methods from a parent class. "
            "3. Polymorphism – same method name behaves differently across different classes. "
            "4. Abstraction – hiding complexity and exposing only relevant functionality."
        ),
        "keywords": [
            "encapsulation", "inheritance", "polymorphism", "abstraction",
            "class", "object", "method", "parent", "child"
        ],
    },
]

# ── BEHAVIORAL QUESTIONS (Pool of 3) ──────────────────────────────────────────

BEHAVIORAL_QUESTIONS = [
    {
        "id": "B1",
        "type": "behavioral",
        "title": "Conflict Resolution",
        "description": (
            "Describe a time when you had a disagreement or conflict with a team member. "
            "How did you handle it? What was the outcome?"
        ),
        "expected_answer": (
            "I once disagreed with a teammate on the technical approach to a feature. "
            "I initiated a one-on-one discussion, actively listened to their perspective, "
            "and we found a compromise that combined the strengths of both approaches. "
            "By staying professional and focusing on the end goal, we resolved it and "
            "built a better solution together."
        ),
        "keywords": [
            "communication", "listening", "compromise", "professionalism",
            "disagreement", "resolution", "team", "outcome"
        ],
    },
    {
        "id": "B2",
        "type": "behavioral",
        "title": "Handling a Challenging Project",
        "description": (
            "What was the **most challenging technical project** you have worked on? "
            "Why was it difficult, and how did you overcome those challenges?"
        ),
        "expected_answer": (
            "My most challenging project involved building a real-time data pipeline under a tight deadline. "
            "The difficulty was optimizing ingestion speed while ensuring data accuracy. "
            "I broke the problem into manageable components, used profiling to find bottlenecks, "
            "and collaborated with the team to implement a queue-based solution. "
            "I overcame the challenge through systematic problem-solving and persistence."
        ),
        "keywords": [
            "challenging", "deadline", "problem solving", "overcame", "collaboration",
            "learning", "adversity", "implemented", "team", "solution"
        ],
    },
    {
        "id": "B3",
        "type": "behavioral",
        "title": "Working Under Pressure",
        "description": (
            "Describe a situation where you had to work under **significant pressure or a tight deadline**. "
            "How did you prioritize your tasks and manage your time?"
        ),
        "expected_answer": (
            "During a product launch, I had to fix a critical bug while also completing feature work. "
            "I prioritized the bug fix first since it was blocking deployment, communicated the situation "
            "to my team, and adjusted the timeline for the feature. "
            "I used task management and focused on what had the most business impact first. "
            "The deployment was successful and on time."
        ),
        "keywords": [
            "priority", "deadline", "pressure", "time management", "communication",
            "focus", "critical", "task", "managed", "delivered"
        ],
    },
]


# ── QUESTION SELECTION LOGIC ───────────────────────────────────────────────────

def select_interview_questions(seed: int = None) -> List[dict]:
    """
    Selects 7 questions for a single interview session.

    Always includes:
      - All 3 coding questions (mandatory)
      - 4 randomly selected from the non-coding pool (4 technical + 3 behavioral = 7 pool)

    Returns a shuffled list of 7 questions.
    """
    if seed is not None:
        random.seed(seed)

    # Non-coding pool: 4 technical + 3 behavioral = 7 total, select 4
    non_coding_pool = TECHNICAL_QUESTIONS + BEHAVIORAL_QUESTIONS
    selected_non_coding = random.sample(non_coding_pool, 4)

    # Combine and shuffle
    session_questions = CODING_QUESTIONS + selected_non_coding
    random.shuffle(session_questions)

    return session_questions
