import sqlite3
import datetime



class UserInfoConnector:
    """
    接続するデータベース
    1. User
        uid : int
        name : str
        like : text
        last : datetime

    2. Comment
        cid : int
        uid : int
        content : text
        reply : text
        time : datetime
    
    """

    def __init__(self, db_path) -> None:
        self.connector = sqlite3.connect(db_path)
        self.cursor = self.connector.cursor()

        self.next_uid = -1
        self.next_cid = -1

        self.format_date = "%Y-%m-%d %H:%M:%S"

        # テーブルの作成(if not exist)
        self.cursor.execute('CREATE TABLE IF NOT EXISTS User(uid int, name varchar(512), like text, last datetime)')

        self.cursor.execute('CREATE TABLE IF NOT EXISTS Comment(cid int, uid int, content text, reply text, last datetime)')
    
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
        self.connector.commit()
    
    # コメントリセット
    def reset_Comment(self):
        sql = 'DROP TABLE IF EXISTS Comment'
        self.cursor.execute(sql)
        self.cursor.execute('CREATE TABLE IF NOT EXISTS Comment(cid int, uid int, content text, reply text, last datetime)')
        self.connector.commit()
    
    def serach_User_by_name(self, user_name:str):
        sql = 'SELECT * FROM User WHERE name="{0}"'.format(user_name)
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        return result
    
    def serach_User_by_uid(self, uid:int):
        sql = 'SELECT * FROM User WHERE uid="{0}"'.format(uid)
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        return result

    # ユーザの検索
    def is_exist_user(self, user_name:str):
        result = self.serach_User_by_name(user_name)

        if len(result)==1:
            return True
        elif len(result)==0:
            return False
        # 長さが2以上 -> 異常
        else:
            print("Number of matched users was more than 1")
            print(result)
            return True
        
    def get_user_id(self, user_name:str):
        result = self.serach_User_by_name(user_name)

        if len(result)==1:
            return int(result[0][0])
        elif len(result)==0:
            return -1
        # 長さが2以上 -> 異常
        else:
            print("Number of matched users was more than 1")
            return -1

    def register_user(self, user_name:str, comment:str, comment_time:datetime.datetime):
        time_str = self.datetime2str(comment_time)
        
        # 次の uid が -1 なら，探索
        if self.next_uid < 0:
            sql = 'SELECT uid FROM User ORDER BY uid' 
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            self.next_uid = int(result[-1][0]) + 1
            
        #  uid, name, like, last 
        sql = "INSERT INTO User VALUES({0}, '{1}', '{2}', '{3}')".format(
            self.next_uid, user_name, "", time_str,
        )
        self.cursor.execute(sql)
        self.connector.commit()

        # 次の番号へ更新
        self.next_uid += 1
    
    # コメント時刻の更新
    def update_comment_time(self, uid:int, comment_time:datetime.datetime):
        sql = "UPDATE User SET last='{0}' WHERE uid={1}".format(self.datetime2str(comment_time), uid)
        self.cursor.execute(sql)
        self.connector.commit()
    
    def register_comment(self, uid:int, comment:str, reply:str, comment_time:datetime.datetime):
        time_str = self.datetime2str(comment_time)
        if self.next_cid < 0:
            sql = 'SELECT cid FROM Comment ORDER BY cid' 
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            self.next_cid= int(result[-1][0]) + 1
        #  cid, uid, content, last 
        sql = "INSERT INTO Comment VALUES({0}, '{1}', '{2}', '{3}', '{4}')".format(
            self.next_cid, uid, comment, reply, time_str,
        )
        self.cursor.execute(sql)
        self.connector.commit()

        # 次の番号へ更新
        self.next_cid += 1
    
    def extract_context(self, uid:int, lst:datetime.datetime):
        time_str = self.datetime2str(lst)
        sql = "SELECT content, reply FROM Comment WHERE last>='{0}' AND uid={1}".format(time_str, uid)
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        return result

    def test_register_User(self):
        self.reset_User()
        self.cursor.execute(
            "INSERT INTO User VALUES(1, '福間創', 'シンセサイザ', '{0}')".format(self.datetime2str(datetime.datetime.now()))
        )
        self.cursor.execute(
            "INSERT INTO User VALUES(2, '平沢進', 'ギター', '{0}')".format(self.datetime2str(datetime.datetime.now()))
        )
        self.cursor.execute(
            "INSERT INTO User VALUES(3, '山田', 'ランニング', '{0}')".format(self.datetime2str(datetime.datetime.now()))
        )
        self.connector.commit()
    
    def test_register_Comment(self):
        """
        2. Comment
            cid : int
            uid : int
            content : text
            reply : text
            time : datetime
        """
        self.reset_Comment()

        self.cursor.execute(
            "INSERT INTO Comment VALUES(1, 1, '平沢さん怒ってます？', '何喧嘩？', '{0}')".format(self.datetime2str(datetime.datetime.now()))
        )
        self.cursor.execute(
            "INSERT INTO Comment VALUES(2, 2, 'それを聞くのは反則だよ', '絶対嘘じゃん！', '{0}')".format(self.datetime2str(datetime.datetime.now()))
        )
        self.connector.commit()