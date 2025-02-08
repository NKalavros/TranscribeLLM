from werkzeug.security import generate_password_hash

# Template for user credentials - DO NOT ADD REAL PASSWORDS HERE
USER_DATABASE = {
    "username": {
        "password": generate_password_hash("your_password_here"),
        "role": "role_here"
    }
}
