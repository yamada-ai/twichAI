# Yamada Dialogue Agent Project(仮)

1. ## はじめに
    - ## 背景
        - 近年，Vtuber と呼ばれる配信者及び配信形態が話題である．
        <!-- - 配信者と視聴者の間を埋める緩衝材として，助手を採用することは有効(?) -->
    - ## 目的

        - 面白そうで他人に自慢できそうな成果物を作りたい
        - おい，お前たちだけで開発しやがって．俺も混ぜろ！


1. ## Agent の構造
    > 本プロジェクトにおいては，エージェントはライブ配信の配信者を想定する．

    > エージェントは次の処理によって配信を行う．

    1. __ユーザ発話の取得__
        - TwitchIOを利用し，チャット欄に書き込まれた各ユーザ発話を取得する
        - ## できそう

    1. __発話意味理解__
        - 取得したユーザ発話 $u_t$ を自然言語処理によって対話行為を識別し，意図及び意味を理解する
        - ユーモアの理解はマジで無理
    
    1. __対話菅理__
        - どーしよーかなー

    1. __発話文生成__
        - 識別した対話行為やその他の技術をアレコレして，ユーザ発話 $u_t$ に対するエージェントの発話$s_t$を生成

    1. __音声合成__
        - $s_t$ をどうにかして音声データへ変換

    1. __エージェント喋ってくれ__
        - どうにかして何かしらのモデルに話してもらおう

    
    - パーフェクトsync
        - VRChat, Vmagic Mirror
        - ボイスを認識して音素(あいうえお)をシンクロ召喚
        - 音声を入力 -> テキストを入力で口を動かす
            - Unity で頑張る
        - テキストから口の形を推定する技術はわからん

    - 音声合成の手法はわからん，どれ？


1. ## 参考文献
    - 雑談対話システムにおける心理的近接発話の戦略が対話の評価に及ぼす影響
        https://www.jstage.jst.go.jp/article/tjsai/36/5/36_36-5_AG21-I/_article/-char/ja/
    - Risky Politeness：対話欲求侵害リスクの高い発話方略によっ て人間らしさを表出する雑談対話システム
        https://www.jstage.jst.go.jp/article/jsaislud/87/0/87_34/_pdf

# 参考文献から得られた知見
- ## 発話戦略
    - Politeness 理論
        - PPS
            - 友達口調や冗談など，リスクを伴う一方で相手との社会的距離を縮める効果のある発話方略
            - 方略の選択を誤った場合，相手のface(対話欲求)の減衰が大きい
        - NPS
            - 敬語や謝罪などの，

# todo
- ## SQL
    - 配信に参加した人の履歴を検索するため
    - Transformer の対話制御
        - 無難な内容しか生成しないのどうにかして！