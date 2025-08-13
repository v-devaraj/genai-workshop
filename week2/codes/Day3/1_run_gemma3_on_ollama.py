# Attempting to run Gemma 3 on Ollama and prompt it with a question.
# Note: Ensure that Ollama is running and the Gemma 3 model is available before executing this script.

import requests
import json

# Connect to Ollama running locally, hence no API key needed.
OLLAMA_URL = "http://localhost:11434/api/generate"

prompt_template =   """
                    [Role]:     You are an experienced AI and NLP practitioner with hands-on expertise in Generative AI, LLMs, and real-world deployments. You have taught professionals and students in this field.

                    [Intent]:   Provide a clear, structured, and practical learning roadmap for mastering Generative AI concepts efficiently, focusing on both theoretical understanding and applied skills.

                    [Context]: 
                                - The learner has a background in Python and basic data analysis.
                                - They are currently exploring AI/ML concepts but want to dive deeper into GenAI, LLMs, prompt engineering, and related tools.
                                - The learner prefers examples, projects, and hands-on learning over purely academic theory.

                    [Instructions]:
                                    1. Structure your answer into clearly labeled sections: 
                                        **Foundational Knowledge**, 
                                        **Core GenAI Concepts**, 
                                        **Hands-On Practice**, 
                                        **Projects & Portfolio**, 
                                        **Advanced Topics**, and 
                                        **Continuous Learning**.
                                    2. In each section, recommend 
                                        **specific resources** (courses, documentation, books, or videos) and 
                                        **practical activities**.
                                    3. Include a **sample weekly schedule** for the first 4 weeks.
                                    4. Suggest at least **3 real-world project ideas** that demonstrate GenAI skills.
                                    5. Provide **common pitfalls** to avoid when learning GenAI.
                                    6. Keep the tone encouraging but professional.

                    [Style & Constraints]:
                                        - Use numbered lists and bullet points where possible.
                                        - Keep explanations concise but actionable.
                                        - Avoid generic advice like “just practice a lot.”
                                        - Focus on **what to do**, **why it matters**, and **how to execute**.

                    [Format]:
                                    Respond in Markdown with:
                                        - Section headings as `##`
                                        - Bullet points and numbered steps
                                        - At the end, include a **Quick Summary** section (max 5 bullet points)
                    """


# Define the question to be asked
question = "How can I learn Generative AI concepts in the most practical and effective way possible?"

# Combine the prompt template with the question
full_prompt = f"{prompt_template}\n\n[Question]: {question}"

# Prepare the payload for the Ollama
payload = {
                "model": "gemma3:1b",
                "prompt": full_prompt,
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": False    #set True of you want streaming response
            }

# Send the request to Ollama
resp = requests.post(OLLAMA_URL, 
                     data = json.dumps(payload), 
                     headers = {"Content-Type": "application/json"})

resp.raise_for_status()  # Raise an error for bad responses
data = resp.json()  # Extract the JSON response

print(data['response'])  # Print the response from Gemma 3
print("\n--- End of Response ---")  # Indicate the end of the response