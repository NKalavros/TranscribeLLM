# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3-venv python3-pip nginx redis-server firewalld git

# Configure firewall
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable

# Create deployment user
sudo adduser deploy
sudo usermod -aG sudo deploy
su - deploy

# Clone your repository
git clone https://github.com/NKalavros/PaperLLM/t
cd academic-summarizer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create .env file
nano .env