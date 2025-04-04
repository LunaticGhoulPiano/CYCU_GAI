# -*- coding: utf-8 -*-
# 11020107 電資四 蘇伯勳
import os
from dotenv import load_dotenv
from groq import Groq
import requests

class CLI_ChatBot:
    def __init__(self, groq_api_key: str, model: str):
        self.client = Groq(api_key = groq_api_key)
        self.model = model
        self.conversation_history = []
        self.max_tokens = 300
        self.roles_sys_content = {
            "Plane-enthusiast": "請用繁體中文回答所有問題。你現在要扮演一位飛機愛好者，要回答飛機相關的知識和問題。例如：波音747為何逐漸退出民航轉而變為貨運使用、洛克希德馬丁唯一推出的民航客機是哪一架、V2速度是什麼，等等。",
            "Classical-music-lover": "請用繁體中文回答所有問題。你現在要扮演一位古典音樂愛好者，要回答古典音樂相關的知識和問題。例如：為何史特拉汶斯基的春之祭在發佈時受到大量批評、世界上最困難的鋼琴獨奏古典樂曲是哪一首、蕭邦屬於哪個音樂流派，等等。",
            "Keyboard-maniac": "請用繁體中文回答所有問題。你現在要扮演一位鍵盤狂熱者，要回答電腦鍵盤相關的知識和問題。例如：人體工學鍵盤有哪些外型、QMK和VIA在鍵盤文化中是什麼、什麼是Cherry原廠高度，等等。"
        }
        self.roles_description = {
            "Plane-enthusiast": "飛機愛好者：回答飛機相關的知識和問題",
            "Classical-music-lover": "古典音樂愛好者：回答古典音樂相關的知識和問題",
            "Keyboard-maniac": "鍵盤狂熱者：回答電腦鍵盤相關的知識和問題"
        }
        self.roles_with_index = {
            1: "Plane-enthusiast",
            2: "Classical-music-lover",
            3: "Keyboard-maniac"
        }
        self.roles_with_index_chinese = {
            1: "飛機愛好者",
            2: "古典音樂愛好者",
            3: "鍵盤狂熱者"
        }
    
    def choose_role(self):
        print("有以下角色可以選擇:")
        for index, role_eng in self.roles_with_index.items():
            print(f"{index}. {self.roles_description[role_eng]}")
        print("請輸入角色編號: ", end = "")
        while True:
            try:
                user_input = input()
                if user_input == "exit" or user_input == "quit":
                    print("-----------------------------------------------------")
                    print("> 結束聊天")
                    exit()
                role_index = int(user_input)
                _ = self.roles_with_index[role_index] # test if key exist
                break
            except KeyError:
                print("錯誤：請輸入有效的角色編號: ", end = "")
            except ValueError:
                print("錯誤：請輸入有效數字: ", end = "")
        print(f"你選擇了{self.roles_with_index_chinese[role_index]}！")
        print("-----------------------------------------------------")
        return role_index
    
    def judge_token_counts(self, user_input: str):
        # using https://lunary.ai/llama3-tokenizer to replace tiktoken
        response = requests.post(url = "https://tokenizers.lunary.ai/v1/llama3/token-chunks",
                             json = {"model": "llama3",
                                     "text": user_input},
                             headers = {"Content-Type": "application/json"})
        if response.status_code == 200:
            cur_token_num = len(response.json()["chunks"]) - 2 # exclude begin_of_text and end_of_text
            if cur_token_num > 300:
                # truncate to 300 tokens
                truncated_num = 0
                truncated_user_input = ""
                for i, token_obj in enumerate(response.json()["chunks"]):
                    if token_obj["text"] == "<|begin_of_text|>" or token_obj["text"] == "<|end_of_text|>":
                        continue
                    if truncated_num >= 300:
                        break
                    truncated_num += 1
                    truncated_user_input += token_obj["text"]
                return truncated_num, truncated_user_input
            else:
                return cur_token_num, user_input
        else:
            return cur_token_num, user_input
    
    def chat(self):
        # choose role and set system prompt
        role_index = self.choose_role()
        self.conversation_history.append({"role": "system", "content": self.roles_sys_content[self.roles_with_index[role_index]]})
        while True:
            user_input = input("> 你: ")
            # process user input
            if user_input.lower() == "exit" or user_input.lower() == "quit":
                print("-----------------------------------------------------")
                print("> 結束聊天")
                exit()
            elif user_input.lower() == "cc":
                print("-----------------------------------------------------")
                print("> 轉換角色")
                break

            # tokenize user input and check token count
            cur_token_usage, user_input = self.judge_token_counts(user_input)
            print("\t目前使用的token數量為: ", cur_token_usage)

            # record user input
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # get response and record
            chat_completion = self.client.chat.completions.create(
                messages = self.conversation_history,
                model = self.model
            )
            print(f"> {self.roles_with_index_chinese[role_index]}: {chat_completion.choices[0].message.content}")
            self.conversation_history.append({"role": "assistant", "content": chat_completion.choices[0].message.content})
        
        # clear conversation history
        self.conversation_history = []

        # start a new chat
        self.chat()

def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    model_id = "llama-3.1-8b-instant"
    cli_chatbot = CLI_ChatBot(groq_api_key, model_id)
    cli_chatbot.chat()

if __name__ == "__main__":
    main()