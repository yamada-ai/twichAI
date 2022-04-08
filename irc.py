from twitchio.ext import commands
from config import *

class Bot(commands.Bot):

    def __init__(self, token, client_id, nick, initial_channels):
        # Initialise our Bot with our access token, prefix and a list of channels to join on boot...
        super().__init__(token, prefix='?' , client_id=client_id,initial_channels=initial_channels)
        
    async def event_ready(self):
        # We are logged in and ready to chat and use commands...
        print(f'Logged in as | {self.nick}')
        print(f'User id is | {self.user_id}')
    
    async def event_message(self, message):
        # Messages with echo set to True are messages sent by the bot...
        # For now we just want to ignore them...
        print(message.author)
        # print(message.channel)
        # print(message.channel.chatters)
        # print(message.)
        if message.author.name == self.nick:
            return

        if message.echo or "Bot :" in message.content:
            return

        # Print the contents of our message to console...
        print(message.author.name+": "+message.content)

        await self.handle_commands(message)

    @commands.command()
    async def hello(self, ctx: commands.Context):
        # Send a hello back!
        await ctx.send(f'Bot :Hello {ctx.author.name}!')

bot = Bot(
    token=TMI_TOKEN,
    client_id=CLIENT_ID,
    nick=BOT_NICK,
    # prefix=BOT_PREFIX,
    initial_channels=CHANNEL,
)

bot.run()