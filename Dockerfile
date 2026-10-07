# Cohort Builder: API + web UI in one container (Hugging Face Spaces, Cloud Run, Fly, any Docker host).
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user api api
COPY --chown=user modules modules
COPY --chown=user utils utils
COPY --chown=user web web

# One worker on purpose: sessions live in this process's memory, and linkage jobs run one at a time.
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
