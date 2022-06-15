format_date = "%Y-%m-%d %H:%M:%S"
from ast import arg
import copy
import json
import datetime
import sqlite3
import argparse

def datetime2str(dt:datetime)->str:
    return dt.strftime(format_date)
    
def str2datetime(time_:str)->datetime:
    return datetime.datetime.strptime(time_, format_date)

# llt : last_load_time
def extract_convs_to_json(llt:str):
    sql = "SELECT content, reply FROM Comment WHERE last>='{0}'".format(llt)
    cursor.execute(sql)
    result = cursor.fetchall()
    json_base = {
        "usr" : "",
        "sys" : ""
    }
    result_json = {"convs":[]}
    for r in result:
        base = copy.copy(json_base)
        
        base["sys"] = r[1]
        base["usr"] = r[0]
        
        result_json["convs"].append(base)
    return json.dumps(result_json, indent=4)

parser = argparse.ArgumentParser()
parser.add_argument("arg1")
parser.add_argument("path")
parser.add_argument("llt")

args = parser.parse_args()

connector = sqlite3.connect(args.path)
cursor = connector.cursor()

# return extract_convs_to_json(args.llt)
