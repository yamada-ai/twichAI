import datetime
import random
from twitchAI.user_info_connector import UserInfoConnector

class Controller:
    def __init__(self):
        self.db_ctrller = UserInfoConnector("./example.db")

        self.live_start_time:datetime.datetime = datetime.datetime.now()
    
    # この関数を呼ぶ
    def reply(self, user_name:str, comment:str, comment_time:datetime.datetime):
        """
        user_name : str
            コメントしたユーザの名前
        comment : str
            コメント内容
        comment_time : datetime.datetime
            コメントした時刻
        """
        utt = ""
        # コメントした人(=ユーザ)が登録されているか
        if self.db_ctrller.is_exist_user(user_name):
            # 初見ではない
            # ユーザのid
            uid = self.db_ctrller.get_user_id(user_name)
            # この配信で初めてのコメントである
            tmp = self.db_ctrller.serach_User_by_uid(uid)
            prev_comment_time = self.db_ctrller.str2datetime(tmp[0][-1])
            # このユーザの最後のコメントが開始時刻よりも前ならば来場初コメント
            delta = self.live_start_time - prev_comment_time
            if int(delta.total_seconds()) >= 0 :
                utt += "{0}さん．こんにちは～．".format(user_name)
                # 前の発話との時差を確認
                tmp = self.db_ctrller.serach_User_by_uid(uid)
                prev_comment_time = self.db_ctrller.str2datetime(tmp[0][-1])
                delta = comment_time - prev_comment_time
                delta_day = int(delta.total_seconds())//(3600*24) 
                # print(int(delta.total_seconds())//(3600*24))
                # 一週間以内
                if delta_day <= 7:
                    # 一日も経ってないよ
                    if delta_day <= 0:
                        utt += "さっきぶり".format(delta_day)
                    else:
                        utt += "{0}日ぶり".format(delta_day)

                    candidate = [
                        "だね．やぁやぁ．",
                        "だね．また来てくれて嬉しいよ．",
                        "．元気？",
                        "にどうも～．",
                        "で合ってる？",
                        ]
                    utt += random.choice(candidate)
                else:
                    # お久しぶりセット
                    candidate = [
                        "お久しぶり！ ",
                        "また会えてうれしいよ！ ",
                        "元気だった？",
                        "生きてたんだぁ・",
                        "おひさ～．",
                        "俺のこと覚えてたんだ．"
                    ]
                    utt += random.choice(candidate)
            
            # 当該ライブでは2回目以降のコメント
            else:
                utt += "2回目以降に反応すると思うなよ"
            
            # 最後の発話時刻を更新
            self.db_ctrller.update_comment_time(uid, comment_time)

        else:
            # 初見である
            utt += "{0}さん，はじめまして！".format(user_name)
            # ユーザを登録
            self.db_ctrller.register_user(user_name, comment, comment_time)
            # ユーザのid
            uid = self.db_ctrller.get_user_id(user_name)


        print(utt)
        # コメントを登録
        self.db_ctrller.register_comment(uid, comment, utt, comment_time)
    