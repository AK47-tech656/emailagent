---
title: Emailagent
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🤖 Autonomous Email Triage Agent (RL-Environment)

This project is an autonomous agent designed to solve the **Email Triage Challenge**. It uses a custom Reinforcement Learning (RL) environment to observe incoming emails and route them to the correct department with an appropriate priority level.

## 🧠 The Brain: Llama-3.3-70B
The agent uses a **Senior Triage System Prompt** that implements strict security and business logic:
* **Security First:** Automatically flags Bitcoin/suspicious links as `spam` with `low` priority.
* **Retention Focus:** Detects "refund" or "legal" threats and routes them to `billing` to prevent customer churn.
* **Infrastructure Alert:** Identifies server outages for high-value senders as `critical support`.

## 🎯 Advanced Reward System (Dense Shaping)
Unlike basic pass/fail agents, this environment uses **Dense Reward Shaping** to provide granular feedback:
* ✅ **+0.5** for Correct Department
* ✅ **+0.4** for Correct Priority
* ✅ **+0.1** for Chain of Thought reasoning (>10 chars)
* ❌ **-0.3** for Wrong Department
* ❌ **-0.2** for Wrong Priority

## 🧪 Chaos Tier Testing
The agent was stress-tested against the "Chaos Tier" dataset, featuring phishing scams, angry customer legal threats, and casual feature requests. 

### Performance Results:
| Email Type | Logic Applied | Reward Score |
| :--- | :--- | :--- |
| **Phishing Scam** | Detected Bitcoin/Suspicious Domain | **1.0 (Perfect)** |
| **Angry Refund Request** | Prioritized Billing over Technical Bug | **0.6 (High)** |
| **Feature Request** | Correctly identified as Low Priority | **1.0 (Perfect)** |

## 🛠️ How to Run (Local Development)
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your token: `$env:HF_TOKEN = "your_token"`
4. Run: `python inference.py`