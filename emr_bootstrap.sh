source /home/hadoop/.bashrc
source /home/hadoop/.bash_profile

cd
mkdir -p Code/research
sudo pip install s3cmd==1.1.0b3

# ----------------------------------------------------------------------
#              Install venv             
# ----------------------------------------------------------------------

aws s3 cp s3://4info-research-e2/lt/bootstrap/spark_virtualenv_hadoop2.tgz .
tar -xzf spark_virtualenv_hadoop2.tgz -C /home/hadoop/

echo export PYSPARK_PYTHON=/home/hadoop/venv/bin/python > ~/.bash_profile
echo export PYSPARK_DRIVER_PYTHON=/home/hadoop/venv/bin/python >> ~/.bash_profile

# ----------------------------------------------------------------------
#              Install Anaconda (Python 3) & Set To Default              
# ----------------------------------------------------------------------

wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh -O ~/Anaconda3-4.2.0-Linux-x86_64.sh
bash ~/Anaconda3-4.2.0-Linux-x86_64.sh -b -p /mnt/anaconda3

echo "----------------------- installation done for anaconda ---------------------------"

export PATH=/mnt/anaconda3/bin:$PATH
echo "export PATH="/mnt/anaconda3/bin:$PATH"" >> ~/.bash_profile

echo export PYSPARK_PYTHON="/mnt/anaconda3/bin/python3" >> ~/.bash_profile
echo export PYSPARK_DRIVER_PYTHON="/mnt/anaconda3/bin/python3" >> ~/.bash_profile

echo "----------------------- Done exporting pyspark path to bash_profile ---------------------------"

# ----------------------------------------------------------------------
#              Install required packages              
# ----------------------------------------------------------------------

pip install pystan==2.18.0.0
echo "-----------------------installation done for pystan---------------------------"

# conda install -c conda-forge -y gcc
conda install -y gcc
echo "-----------------------installation done for gcc---------------------------"

conda install -c conda-forge -y fbprophet
cho "-----------------------installation done for fbprophet---------------------------"


conda install -c intel -y scikit-learn
echo "-----------------------installation done for scikit-learn---------------------------"

conda install -y pandas
echo "-----------------------installation done for pandas---------------------------"

