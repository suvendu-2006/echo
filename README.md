# Echo

A private, password-gated AI assistant. Deployed on Koyeb (free tier, no card).

Secrets (set as Koyeb environment variables, never in code):
- ECHO_PASSWORD - your login password
- GROQ_API_KEY - your Groq brain
- ECHO_HOST=0.0.0.0 - required inside the container
