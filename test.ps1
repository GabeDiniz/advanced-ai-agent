# List models (confirms Ollama is really on 11434)
Invoke-RestMethod http://127.0.0.1:11434/api/tags

# List model names
(Invoke-RestMethod http://127.0.0.1:11434/api/tags).models.name
