from twitchio.ext import commands
import sys
from config import *
import datetime
from twitchAI.controller import Controller

class Bot(commands.Bot):

    def __init__(self, token, client_id, nick, initial_channels):
        # Initialise our Bot with our access token, prefix and a list of channels to join on boot...
        super().__init__(token, prefix='?' , client_id=client_id,initial_channels=initial_channels)

        self.ctrler = Controller()
        
    async def event_ready(self):
        # We are logged in and ready to chat and use commands...
        print(f'Logged in as | {self.nick}')
        print(f'User id is | {self.user_id}')
    
    async def event_message(self, message):
        # Messages with echo set to True are messages sent by the bot...
        # For now we just want to ignore them...
        print(message.author)
        if message.author.name == self.nick:
            return

        if message.echo or "Bot :" in message.content:
            return

        # Print the contents of our message to console...
        print(message.author.name+": "+message.content)

        # await ctx.send(f'Bot :Hello {ctx.author.name}!')

        # ここから発話を生成
        # 発話した時刻を取得
        user_name = message.author.name
        comment = message.content
        comment_time = datetime.datetime.now()
        self.ctrler.reply(user_name, comment, comment_time)
        await self.handle_commands(message)

    @commands.command()
    async def hello(self, ctx: commands.Context):
        # Send a hello back!
        await ctx.send(f'Bot :Hello {ctx.author.name}!')
    

if __name__ == '__main__':
    bot = Bot(
        token=TMI_TOKEN,
        client_id=CLIENT_ID,
        nick=BOT_NICK,
        # prefix=BOT_PREFIX,
        initial_channels=CHANNEL,
    )

    bot.run()