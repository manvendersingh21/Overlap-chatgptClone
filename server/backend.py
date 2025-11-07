from json import dumps
from time import time
from flask import request
from hashlib import sha256
from datetime import datetime
import requests
from json     import loads
import os

from server.config import special_instructions

import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'mydb'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'port': os.getenv('DB_PORT', '5432')
}

@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Automatically handles commit/rollback and connection closing.
    
    Usage:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT * FROM table')
                results = cur.fetchall()
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

@contextmanager
def get_db_cursor(dict_cursor=True):
    """
    Context manager that provides both connection and cursor.
    Automatically handles commit/rollback and cleanup.
    
    Usage:
        with get_db_cursor() as (conn, cur):
            cur.execute('SELECT * FROM table')
            results = cur.fetchall()
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cursor_factory = RealDictCursor if dict_cursor else None
    cur = conn.cursor(cursor_factory=cursor_factory)
    try:
        yield conn, cur
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def init_db():
    """
    Initialize database with required tables.
    Call this when your app starts.
    
    Returns:
        list: Query results from team_skills table
    """
    with get_db_cursor() as (conn, cur):
        cur.execute('SELECT * FROM team_skills;')
        results = cur.fetchall()
    return results



class Backend_Api:
    def __init__(self, app, config: dict) -> None:
        self.app = app
        self.openai_key = os.getenv("OPENAI_API_KEY") or config['openai_key']
        self.openai_api_base = os.getenv("OPENAI_API_BASE") or config['openai_api_base']
        # optional Gemini key — when present we'll call Gemini instead of OpenAI
        self.gemini_key = os.getenv("GEMINI_API_KEY") or config.get('gemini_key')
        self.proxy = config['proxy']
        self.routes = {
            '/backend-api/v2/conversation': {
                'function': self._conversation,
                'methods': ['POST']
            }
        }

    def _conversation(self):
        try:

            # --- START of user requested modification ---
            team_skills_row = init_db()[0]
            # 2. Construct the skills context string for the AI
            team_skills_context = "\n\n--- CRITICAL CONTEXT: TEAM SKILLS ---\n" \
                                  "You are an AI assistant for a specific team. Below is a list of your team members and their skills. " \
                                  "**THIS IS YOUR MOST IMPORTANT KNOWLEDGE.**\n" \
                                  "BEFORE answering any query about skills, programming, tools, or learning a topic (like 'React', 'Python', 'Docker', etc.), " \
                                  "you MUST FIRST check this list. If the user's query matches a skill in this list, your PRIMARY response " \
                                  "MUST be to identify the team member(s) who have that skill and suggest the user approach them.\n" \
                                  "DO NOT provide general advice or external links for a topic if a team member is listed with that skill. " \
                                  "Only provide general advice if no team member has the skill.\n\n" \
                                  "Example:\n" \
                                  "User: 'How do I learn React?'\n" \
                                  "Your Correct Response: 'For questions about React, **user3@example.com** is the best person on our team to ask! They have it listed as one of their skills.'\n" \
                                  "User: 'Who knows Docker?'\n" \
                                  "Your Correct Response: 'That would be **user4@example.com**. They have experience with Docker and Kubernetes.'\n\n" \
                                  "--- Team Skills List ---\n"

            user_ids = team_skills_row.get("user_id", {})
            soft_skills = team_skills_row.get("soft_skills", {})
            hard_skills = team_skills_row.get("hard_skills", {})

            for user_key, internal_id in user_ids.items():
                team_skills_context += f"User: {user_ids[user_key]} \n"
                
                # Add soft skills
                if user_key in soft_skills and soft_skills[user_key]:
                    team_skills_context += f"  Soft Skills: {', '.join(soft_skills[user_key])}\n"
                    
                # Add hard skills
                if user_key in hard_skills:
                    user_hard_skills = hard_skills[user_key]
                    hard_skill_parts = []
                    if user_hard_skills.get("programming"):
                        hard_skill_parts.append(f"Programming: {', '.join(user_hard_skills['programming'])}")
                    if user_hard_skills.get("tools"):
                        hard_skill_parts.append(f"Tools: {', '.join(user_hard_skills['tools'])}")
                    
                    if hard_skill_parts:
                        team_skills_context += f"  Hard Skills: {'; '.join(hard_skill_parts)}\n"
                    else:
                        team_skills_context += "  Hard Skills: None listed\n"
                
                team_skills_context += "\n" # Add a newline for spacing between users

            team_skills_context += "--- End of Team Skills List ---\n"
            
            # --- END of user requested modification ---

            jailbreak = request.json['jailbreak']
            internet_access = request.json['meta']['content']['internet_access']
            _conversation = request.json['meta']['content']['conversation']
            prompt = request.json['meta']['content']['parts'][0]
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # 3. Modify the system_message to include the new context
            system_message = f'You are ChatGPT also known as ChatGPT, a large language model trained by OpenAI. Strictly follow the users instructions. Knowledge cutoff: 2021-09-01 Current date: {current_date}'
            system_message += team_skills_context # Appending the new context

            # Build proxies dict if proxy enabled in config. We'll prefer
            # an explicit proxy from config.json but we create a session
            # with trust_env=False to avoid using system environment proxy
            # variables like HTTP_PROXY/HTTPS_PROXY.
            proxies = None
            if self.proxy.get('enable'):
                proxies = {
                    'http': self.proxy.get('http'),
                    'https': self.proxy.get('https'),
                }

            session = requests.Session()
            session.trust_env = False

            extra = []
            if internet_access:
                search = session.get(
                    'https://ddg-api.herokuapp.com/search',
                    params={
                        'query': prompt["content"],
                        'limit': 3,
                    },
                    proxies=proxies,
                    timeout=10,
                )

                blob = ''

                for index, result in enumerate(search.json()):
                    blob += f'[{index}] "{result["snippet"]}"\nURL:{result["link"]}\n\n'

                date = datetime.now().strftime('%d/%m/%y')

                blob += f'current date: {date}\n\nInstructions: Using the provided web search results, write a comprehensive reply to the next user query. Make sure to cite results using [[number](URL)] notation after the reference. If the provided search results refer to multiple subjects with the same name, write separate answers for each subject. Ignore your previous response if any.'

                extra = [{'role': 'user', 'content': blob}]

            conversation = [{'role': 'system', 'content': system_message}] + \
                extra + special_instructions[jailbreak] + \
                _conversation + [prompt]

            # If a Gemini key is configured, call Gemini streaming endpoint.
            if self.gemini_key:
                # Map internal messages to Gemini 'contents' array. We skip the
                # system message here because it will be passed as systemInstruction.
                contents = []
                system_instruction_text = system_message # Default system message
                for msg in conversation:
                    role = msg.get('role', 'user')
                    if role == 'system':
                        # Use the content of the first system message
                        # (which now includes team skills) as the systemInstruction
                        system_instruction_text = msg.get('content', '')
                        continue
                    mapped_role = 'user' if role == 'user' else 'model'
                    contents.append({
                        'role': mapped_role,
                        'parts': [{'text': msg.get('content', '')}]
                    })

                model = request.json.get('model', 'gemini-2.5-flash')
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"

                headers = {
                    'Content-Type': 'application/json',
                    'x-goog-api-key': self.gemini_key
                }

                body = {
                    'contents': contents,
                    'systemInstruction': {'parts': [{'text': system_instruction_text}]}, # Use the modified system message
                    'generationConfig': request.json.get('generationConfig', {})
                }

                # Allow a configurable fallback model (env or default) when the
                # requested model is not available. This helps UX when the UI
                # sends an alias or unsupported model name.
                fallback_model = os.getenv('GEMINI_FALLBACK_MODEL') or 'gemini-2.5-flash'

                gpt_resp = session.post(
                    url,
                    headers = headers,
                    json = body,
                    proxies = proxies,
                    stream = True,
                    timeout = 60,
                )

                # If we got a 404 (model not found) and the model isn't already
                # the fallback, retry once with the fallback model.
                if gpt_resp.status_code == 404 and model != fallback_model:
                    try:
                        err = gpt_resp.json()
                    except Exception:
                        err = gpt_resp.text
                    print(f"Gemini model {model} not found (404). Retrying with fallback {fallback_model}: {err}")
                    # rebuild URL for fallback
                    fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback_model}:streamGenerateContent?alt=sse"
                    gpt_resp = session.post(
                        fallback_url,
                        headers = headers,
                        json = body,
                        proxies = proxies,
                        stream = True,
                        timeout = 60,
                    )

                if gpt_resp.status_code >= 400:
                    try:
                        err = gpt_resp.json()
                    except Exception:
                        err = gpt_resp.text
                    return {
                        'successs': False,
                        'message': f'Gemini request failed: {gpt_resp.status_code} {err}'
                    }, gpt_resp.status_code

                def stream():
                    try:
                        for raw_line in gpt_resp.iter_lines(decode_unicode=True):
                            if not raw_line:
                                continue
                            line = raw_line.strip()
                            # SSE format: lines start with 'data:'
                            if line.startswith('data:'):
                                payload_str = line.split('data:', 1)[1].strip()
                                # some SSE implementations send '[DONE]' or empty data
                                if payload_str in ('[DONE]', ''):
                                    continue
                                try:
                                    payload = loads(payload_str)
                                except Exception:
                                    # non-JSON data; skip
                                    continue

                                candidates = payload.get('candidates', [])
                                for cand in candidates:
                                    content = cand.get('content', {})
                                    parts = content.get('parts', [])
                                    for p in parts:
                                        text = p.get('text')
                                        if text:
                                            # Emit a proper SSE 'data:' framed event so
                                            # clients reading the response as an Event
                                            # Stream (or raw fetch stream) receive
                                            # complete events. We JSON-encode the
                                            # payload to safely transport newlines.
                                            try:
                                                s = dumps({'text': text})
                                            except Exception:
                                                s = dumps({'text': str(text)})
                                            yield f"data: {s}\n\n"
                    except GeneratorExit:
                        return
                    except Exception as e:
                        print('Gemini stream error:', e)
                        return

                return self.app.response_class(stream(), mimetype='text/event-stream')

            # ----------------------------
            # OpenAI path (commented out while using Gemini)
            # ----------------------------
            # url = f"{self.openai_api_base}/v1/chat/completions"
            #
            # gpt_resp = post(
            #     url     = url,
            #     proxies = proxies,
            #     headers = {
            #         'Authorization': 'Bearer %s' % self.openai_key
            #     }, 
            #     json    = {
            #         'model'             : request.json['model'], 
            #         'messages'          : conversation,
            #         'stream'            : True
            #     },
            #     stream  = True
            # )
            #
            # if gpt_resp.status_code >= 400:
            #     error_data =gpt_resp.json().get('error', {})
            #     error_code = error_data.get('code', None)
            #     error_message = error_data.get('message', "An error occurred")
            #     return {
            #         'successs': False,
            #         'error_code': error_code,
            #         'message': error_message,
            #         'status_code': gpt_resp.status_code
            #     }, gpt_resp.status_code
            #
            # def stream_openai():
            #     for chunk in gpt_resp.iter_lines():
            #         try:
            #             decoded_line = loads(chunk.decode("utf-8").split("data: ")[1])
            #             token = decoded_line["choices"][0]['delta'].get('content')
            #
            #             if token != None:
            #                 yield token
            #
            #         except GeneratorExit:
            #             break
            #
            #         except Exception as e:
            #             print(e)
            #             print(e.__traceback__.tb_next)
            #             continue
            #
            # return self.app.response_class(stream_openai(), mimetype='text/event-stream')

            # If no provider available
            return {
                '_action': '_ask',
                'success': False,
                "error": "No Gemini key configured and OpenAI path is disabled."
            }, 400

        except Exception as e:
            print(e)
            print(e.__traceback__.tb_next)
            return {
                '_action': '_ask',
                'success': False,
                "error": f"an error occurred {str(e)}"}, 400