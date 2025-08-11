import io
from openai import OpenAI
import math
import json
import jsonlines
from http import client
from random import randint
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def read_json_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as r_file:
        datas = json.load(r_file)
    r_file.close()
    return datas


def read_json_file_line(file_path):
    datas = []
    with io.open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line)
            datas.append(json_object)
    return datas


def save_file(datas, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(datas, file, indent=4, ensure_ascii=False)
    file.close()


def write_file(data, file_name):
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(json.dumps(data))
        file.write('\n')
    file.close()


def read_jsonl_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        datas = list(jsonlines.Reader(file))
    file.close()
    return datas


def haversine(coord1, coord2):
    """
    计算两个地理坐标点之间的Haversine距离（单位：千米）

    参数:
    coord1 -- 第一个地点的坐标元组（经度, 纬度）
    coord2 -- 第二个地点的坐标元组（经度, 纬度）

    返回:
    两点之间的千米数（浮点数）
    """
    # 地球平均半径（单位：千米）
    R = 6371.0

    # 解析坐标（经度在前，纬度在后）
    lon1, lat1 = coord1
    lon2, lat2 = coord2

    # 将度转换为弧度
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # 计算经度和纬度的差值
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    # Haversine公式计算
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # 计算距离
    distance = R * c

    return distance


# def ask_gpt(prompts: str, model, url, api_key):
#     client = OpenAI(
#         base_url=url,
#         api_key=api_key
#     )
#
#     response = client.chat.completions.create(
#       model=model,
#       messages=[
#         {"role": "user", "content": prompts},
#
#       ]
#     )
#     response = response.choices[0].message.content
#     return response


def multi_thread_ask_gpt(api_key, model, url, prompts_list, save_file_path):

    # 使用 ThreadPoolExecutor 创建线程池
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交任务到线程池
        future_to_prompt = {
            executor.submit(ask_gpt_with_except, prompt, model, url, api_key): prompt
            for prompt in prompts_list
        }

        # 使用 tqdm 显示进度条
        with tqdm(total=len(prompts_list), desc="Processing prompts") as pbar:
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    data = {"prompt": prompt, "result": result}
                    write_file(data, save_file_path)
                except Exception as e:
                    data = {"prompt": prompt, "error": str(e)}
                    write_file(data, save_file_path)
                finally:
                    pbar.update(1)


    print(f"Results saved to {save_file_path}")


def ask_gpt_with_except(prompts, model, url, api_key):
    user_agents = [
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
        "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
        "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
        "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
        "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
        "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
        "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
    ]
    random_agent = user_agents[randint(0, len(user_agents) - 1)]

    try:
        conn = client.HTTPSConnection(url)
        payload = json.dumps({
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompts}]
        })
        headers = {
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + api_key,
            'User-Agent': random_agent,
            'Content-Type': 'application/json'
        }

        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        response = res.read()
        response_utf8 = response.decode('utf-8')
        json_llm_answers = json.loads(response_utf8)
        llm_answers = json_llm_answers['choices'][0]['message']['content']
        return {"status": "success", "response": llm_answers}
    except Exception as e:
        return {"status": "error", "error": str(e)}