
# Structured DDx Agent 🩺

> **A Human-in-the-Loop clinical reasoning agent. Implements hypothetico-deductive logic for structured differential diagnosis (DDx) using LangGraph.**

This project demonstrates an agentic workflow for medical triage that moves beyond simple chatbots. Instead of generating linear text, the agent actively builds a probabilistic differential diagnosis, identifies clinical ambiguity, and generates targeted screening questions to rule conditions in or out.

## 🧠 Clinical Logic Architecture

The system mimics the **Hypothetico-Deductive Reasoning** process used by clinicians:

1.  **Initial Triage:** Collects presenting symptoms and history.
2.  **Differential Diagnosis (DDx):** Generates a structured list of potential conditions, ranked by probability and severity (e.g., "Life Threatening" vs. "Mild").
3.  **Refinement Loop:** The core reasoning engine. It identifies the condition with the highest uncertainty and generates *one* specific question to test that hypothesis.
4.  **Human-in-the-Loop (HITL):** Using LangGraph's `interrupt` mechanism, the system pauses execution to validate input with the user, ensuring safety and preventing hallucinations.

### System Flow

```mermaid
graph TD
    A["Patient Input: 'Headache'"] --> B{"Initial Triage AI"}
    B -->|Generate| C["Screening Questions"]
    C --> D(("Human-in-the-Loop"))
    D -->|Patient Answers| E["Generate Differential Diagnosis (DDX)"]
    
    subgraph RefinementLoop ["Clinical Refinement Loop"]
    E --> F{"High Confidence OR<br/>Safety Threshold Met?"}
    F -- No --> G["Identify Top Ambiguity"]
    G --> H["Generate Targeted<br/>Rule-In/Rule-Out Question"]
    H --> I(("Human-in-the-Loop"))
    I -->|New Evidence| J["Refine Probabilities & Severity"]
    J --> F
    end
    
    F -- Yes --> K["Final Clinical Summary<br/>& Triage Advice"]
    
    style D fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
````

### Graph Topology

![Graph Topology](images/graph_topology.png)

## 🛠️ Tech Stack

  * **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) (Cyclic state management)
  * **LLM:** OpenAI GPT-5.1
  * **Validation:** [Pydantic](https://www.google.com/search?q=https://docs.pydantic.dev/) (Strict schemas for Clinical Data)
  * **Pattern:** Human-in-the-Loop (HITL) via state checkpoints

## 🚀 Key Features

  * **Cyclic Reasoning:** The graph loops dynamically based on the confidence of the diagnosis, simulating a doctor "thinking" through a case.
  * **Structured Output:** The agent does not return strings; it returns strict Python objects for reliability:
    ```python
    class Diagnosis(BaseModel):
        condition: str
        probability: float
        severity: str  # e.g., "life_threatening"
        reasoning: str
    ```
  * **Safety Guardrails:** Emergency conditions (red flags) are tracked explicitly in the state metadata.

## 📦 Quick Start

### 1\. Clone the repository

```bash
git clone https://github.com/navidmovafeghi/structured-ddx-agent.git
cd structured-ddx-agent
```

### 2\. Set up environment

Create a `.env` file in the root directory and add your OpenAI API key:

```text
OPENAI_API_KEY=sk-your-key-here
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

### 4\. Run the agent

```bash
python src/graph.py
```

## 🎯 Project Purpose: Closing the "Symptom-to-Care" Gap

**This tool is designed to be an intelligent notifier, not a replacement for a doctor.**

In the current healthcare landscape, there is often a critical time gap between the onset of symptoms and professional medical evaluation. Many patients underestimate serious symptoms due to a lack of medical knowledge, leading to delayed treatment and poorer outcomes. Conversely, anxiety can lead to unnecessary ER visits for benign conditions.

The goal of this project is to bridge that gap by providing an **accessible, structured triage layer**:

* **📉 Reduce Time-to-Care:** Help patients identify concerning "Red Flag" signals immediately, motivating them to seek professional care faster.
* **🧠 Empower Patients:** Translate vague feelings ("I feel weird") into structured clinical data ("Sudden onset, severe severity, associated with nausea") that is useful for doctors.
* **🚨 Early Detection:** Catch potential life-threatening conditions (e.g., Meningitis, Stroke signs) that a layperson might miss, serving as an early warning system.

**What this is NOT:**

* It is **not** an automated diagnostic device.
* It does **not** prescribe medication or treatment.
* It is **not** a substitute for professional medical advice.

## ⚠️ Disclaimer

**This is an architectural demonstration of AI reasoning patterns, not a medical device.**

It is designed to showcase how LangGraph can model complex decision-making loops in healthcare. It should **not** be used for actual medical diagnosis, triage, or treatment. Always consult a qualified healthcare provider for medical issues.


