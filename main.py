
from twitchAI import user_info_connector

if __name__ == '__main__':
    db_path = "DB/info.db"
    connector = user_info_connector.UserInfoConnector(db_path)