from helium import *
import time
import keyring

def set_login() -> dict:
    service = input("Enter service name: ")
    user = input("Enter username: ")
    password = input("Enter password: ")
    keyring.set_password(service, user, password)
    return {"username": user, "password": password}

def autorefresh(url: str, interval: int = 60, headless: bool = False, credentials: dict = None, verbose: bool = True) -> None:
    try:
        start_firefox(url=url, headless=headless)
        while True:
            if Text("Login").exists() or Text("Username").exists() or Text("Password").exists():
                if verbose:
                    print("Logging in...")
                if credentials is None:
                    if verbose:
                        print("No credentials provided. Setting new credentials...")
                    credentials = set_login()
                    if verbose:
                        print("Credentials stored in keyring.")
                write(credentials["username"], into="Username")
                write(credentials["password"], into="Password")
                del credentials
                if Button("Login").exists():
                    click("Login")
                elif Button("Sign In").exists():
                    click("Sign In")
                time.sleep(5)
            if verbose:
                print("Logged in.")
            time.sleep(interval)
            refresh()
            if verbose:
                print(f"Page refreshed at {time.strftime('%H:%M:%S')}")
    except Exception as e:
        print(e)
        kill_browser()