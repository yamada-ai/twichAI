import sqlite3
import datetime



class UserInfoConnector:
    """
    接続するデータベース
    1. User
        uid : int
        name : str
        last : datetime
        like : text

    2. Comment
        cid : int
        uid : int
        content : text
        time : datetime
    
    """

    def __init__(self, db_path) -> None:
        self.connector = sqlite3.connect(db_path)
        self.cursor = self.connector.cursor()

        self.format_date = "%Y-%m-%d %H:%M:%S"

        # テーブルの作成(if not exist)
        self.cursor.execute('CREATE TABLE IF NOT EXISTS User(uid int, name varchar(512), like text, last datetime)')

        self.cursor.execute('CREATE TABLE IF NOT EXISTS Comment(cid int, uid int, content text, last datetime)')
    
    def datetime2str(self, dt:datetime)->str:
        return dt.strftime(self.format_date)
    
    def str2datetime(self, time:str)->datetime:
        return datetime.datetime.strptime(time, self.format_date)

    # テーブルリセット
    def reset_infomation(self):
        self.reset_User()
        self.reset_Comment()

    # ユーザリセット
    def reset_User(self):
        sql = 'DROP TABLE IF EXISTS User'
        self.cursor.execute(sql)
        sql = 'CREATE TABLE IF NOT EXISTS User(uid int, name varchar(512), like text, last datetime)'
        self.cursor.execute(sql)
    
    # コメントリセット
    def reset_Comment(self):
        sql = 'DROP TABLE IF EXISTS Comment'
        self.cursor.execute(sql)
        sql = 'CREATE TABLE IF NOT EXISTS Comment(cid int, uid int, content text, last datetime)'
        self.cursor.execute(sql)
    
    # ユーザの検索
    # def search_(self, name:str):
    #     sql = 'SELECT * FROM User2 WHERE name="{0}"'.format(name)
    #     c.execute(q)
    #     result = c.fetchall()
    #     return result