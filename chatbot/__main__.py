from . import mcmaster_chat_bot


def main():
    mcmaster_chat_bot.start()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
