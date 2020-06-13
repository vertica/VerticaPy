<!DOCTYPE html>
<html>
    <?php include('include/head.php'); ?>
    <body>
      <div><?php include('include/header.php'); ?></div>
      <div id="content">
        <div><?php include('include/logo.php'); ?></div>
        <h1>Installation</h1>
        <p>
          Setting up the environment is crucial. That's why each step will be well detailed. To be able to enjoy the most of Vertica ML Python, let's see how to:
          <ul class="ul_content">
            <a href="#title_1"><li>Install Vertica</li></a>
            <a href="#title_2"><li>Making the VM IP address static - <b>Optional</b></li></a>
            <a href="#title_3"><li>Create a DataBase DSN (Data Source Name) - <b>Optional</b></li></a>
            <a href="#title_4"><li>Install Python3</li></a>
            <a href="#title_5"><li>Install Jupyter - <b>Optional</b></li></a>
            <a href="#title_6"><li>Install Vertica ML Python</li></a>
          </ul>
        </p>
        <p>If you already have Vertica and Python3 installed in your environment. To install Vertica ML Python, just run the following command.</p><br>
        <div class="highlight"><pre><span></span>pip3 install vertica_ml_python</pre></div>
        <h2 id="title_1">Vertica Installation</h2>
        <p>Vertica is the most advanced analytics data warehouse based on a massively scalable architecture with the broadest set of analytical functions spanning event and time series, pattern matching, geospatial and end-to-end in-database machine learning. Vertica enables you to easily apply these powerful functions to the largest and most demanding analytical workloads, arming you and your customers with predictive business insights faster than any analytics data warehouse in the market.</p><br>
        <p>Vertica provides a unified analytics platform across major public clouds and on-premises data centers and integrates data in cloud object storage and HDFS without forcing you to move any of your data.</p><br>
        <p>To learn more about the Vertica DataBase, you can go to the <a href="https://www.vertica.com/about/">Vertica Official Website</a>.</p><br>
        <p>If you already have Vertica installed in your Machine, you can skip this step. Otherwise you have 3 options to get Vertica for free.</p>
        <p><ul class="ul_content">
            <li>You have a machine where Linux is installed and you want to install Vertica Community Edition directly on your Machine. Please look at <a href="https://www.youtube.com/watch?v=D5SbzVVR_Ps">this video</a>. Tim Donar will explain you how to do it. You can also use the same process if you want to install Linux from scratch in a VM. By choosing this option, you'll be able to get the most of your machine as you'll customize the installation parameters. However, it is much more complicated than the two next options.</li>
            <li>You decide to use the Vertica Community Edition VM. In this case, follow the instructions of the <a href="https://www.vertica.com/docs/VMs/Vertica_CE_VM_Download_and_Startup_Instructions.pdf">Vertica Community Edition Virtual Machine Installation Guide</a>.</li>
            <li>You decide to use a Vertica machine remotely by subscribing to the <a href="https://academy.vertica.com/">Vertica Academy</a> program. It is free and you can get a remote access to a machine by sending a mail to the <a href="https://academy.vertica.com/path-player?courseid=essentials9x&unit=5e41c0fdc1ba3177028b4584Unit">Vertica Academy team</a>.</li>
          </ul>
        </p>

       <h2 id="title_2">Making the VM IP address static - Optional</h2>

        <p>This step is essential if you want to work from your own OS environment rather than directly in the VM. To be able to create a bridge between your OS and your VM. To create a communication between your OS and the VM. Open the parameters of your VM and create a second adapter (Adapter 2) and set it to bridged adapter.</p>
        <br>
        <center>
          <img src="img/network/0.png" width="30%">
          <img src="img/network/1.png" width="30%">
          <img src="img/network/2.png" width="30%">
        </center> 
        <br>
        <p>
          Look at the IP address of your machine using the <b>ifconfig</b> command in your Terminal.
        </p>
        <br>
        <center>
          <img src="img/network/ifconfig.png" width="60%">
        </center>
        <br>
        <p>As my local inet is 192.168.1.8, let's pick 192.168.1.150 as static IP address for my VM (I could choose 192.168.1.x as long as it is not taken by another machine). Let's make our VM IP address static. Go to the network configuration of your VM and change the IPV4.</p>
        <br>
        <center>
          <img src="img/network/3.png" width="30%">
          <img src="img/network/4.png" width="30%">
          <img src="img/network/5.png" width="30%">
        </center> 
        <br>
        <p>Turn the connection off and on.</p>
        <center>
          <img src="img/network/6.png" width="45%">
          <img src="img/network/7.png" width="45%">
        </center>
        <br>
        <p>You can now forget your VM as long as you turn it on and you are connected to a network (even one which does not work is enough to make your machine believe that it is connected). From now, you can control remotely your VM using an SSH to dbadmin@192.168.1.150</p>

        <h2 id="title_3">Create a DataBase DSN (Data Source Name) - Optional</h2>

        <p>Creating DSN can make you win a lot of time when working with DataBases. To create a DSN, you can follow the instructions in the <a href="https://www.vertica.com/docs/9.2.x/HTML/Content/Authoring/ConnectingToVertica/InstallingDrivers/CreatingAnODBCDataSourceNameDSN.htm?tocpath=Connecting%20to%20Vertica%7CClient%20Libraries%7CCreating%20an%20ODBC%20Data%20Source%20Name%20(DSN)%7C_____0">Vertica Website</a>.</p><br>

        <p>As the DSN configuration could be a little bit more complicated in MACOS due to some updates. Let's see the different steps that you may have to do.</p><br>

        <p>Vertica includes its own drivers for different platforms, included MacOS. Just download the drivers from the <a href="https://my.vertica.com/download/vertica/client-drivers/">Vertica Website</a>.</p><br>

        <p>Select your version of Vertica and download the corresponding pkg package to install in your operating system. Vertica client drivers have backwards compatibility since version 8.1.0. So, for example, it is ok if we install version 9 client drivers and connect to 8.1.1 Vertica database. Once the pkg file is dowloaded it can be installed on the system by double clicking in the file. Be aware that the package is downloaded from an unknown developer, so MacOS is going to ask for permission to install it. To do this just go to System Preferences > Security and Privacy > General and accept the installation.</p><br>
        <p>If we didn't change the location of the installation, the new Vertica drivers should be installed in folder "lib" located at "/Library/Vertica/ODBC/lib". Location can be checked in the Terminal.</p>
        <br>
        <div class="highlight"><pre><span></span>root@ubuntu:~$ ls /Library/Vertica/ODBC/lib/</pre></div>
        <br>
        <p>There are two files:</p>
        <ul class="ul_content">
        <li><b>libverticaodbc.dylib</b> - This is the driver</li>
        <li><b>vertica.ini</b> - Vertica configuration file template for the driver</li>
        </ul>
        <p>Once Vertica driver is installed, next step is to customize configuration files in order to use DB DSN, encodings and drivers. The following just need to be configured.
        <ul class="ul_content">
          <li> /Library/Vertica/ODBC/lib/vertica.ini</li>
          <li>~/Library/ODBC/odbc.ini</li>
          <li> ~/Library/ODBC/odbcinst.ini</li>
          <li>VERTICAINI and ODBCINI system parameters</li>
        </ul>
        <p>The easiest way to do this is to use ODBC Administrator, a GUI Tool to configure the odbc.ini and odbcinst.ini
        The ODBC Administrator can be downloaded at the <a href="https://support.apple.com/kb/DL895?locale=en_US">Apple Website</a>.</p>

        <p>Once downloaded and installed, you can add a DSN by double clicking on ODBC Administrator application (in Applications/Utilities/ODBC Administrator) and just do as following.</p><br>

        <center>
          <img src="img/DSN/1.png" width="30%">
          <img src="img/DSN/2.png" width="30%">
          <img src="img/DSN/3.png" width="30%">
          <img src="img/DSN/4.png" width="30%">
          <img src="img/DSN/5.png" width="30%">
          <img src="img/DSN/6.png" width="30%">
        </center><br>

        <p>We check then that odbc.ini has the DSN added previously.</p><br>
        <div class="highlight"><pre><span></span>root@ubuntu:~$ cat ~/Library/ODBC/odbc.ini

# Output
[ODBC Data Sources]
VMart = Vertica
MLTesting = Vertica
MyMLTesting = Vertica

[ODBC]
Trace = 0
TraceAutoStop = 0
TraceFile =
TraceLibrary =
?.

[MLTesting]
Driver = /Library/Vertica/ODBC/lib/libverticaodbc.dylib
Description = MLTesting on Azure
Servername = vazure
Database = MLTesting
UID = dbadmin
PWD =
Port = 5433</pre></div><br>

        <p>The Vertica ini file is already configured by default, but I recommend to copy it to the same place as the other odbc files.</p><br>
        <div class="highlight"><pre><span></span>root@ubuntu:~$ cp /Library/Vertica/ODBC/lib/vertica.ini ~/Library/ODBC/

# Change the encoding parameter from UTF-32 to UTF-16 in the copied vertica.ini.
[Driver]
ErrorMessagesPath=/Library/Vertica/ODBC/messages/
ODBCInstLib=/usr/lib/libiodbcinst.dylib
DriverManagerEncoding=UTF-16</pre></div><br>

        <p>Last step is to add the environment variables VERTICAINI and ODBCINI. Just write them into your bash_profile (or /etc/profile for system wide).</p><br>

        <div class="highlight"><pre><span></span>root@ubuntu:~$ vim ~/.bash_profile

## Adding some parameters to work with ODBC for Vertica
[?]
export ODBCINI=/Library/ODBC/odbc.ini
export VERTICAINI=/Library/ODBC/vertica.ini
[?]</pre></div>

        <h2 id="title_4">Install Python3</h2>

        <p>Installing Python3 is as easy as downloading a file. Just follow the instructions in the <a href="https://www.python.org/downloads/">Python Website.</a></p>

        <h2 id="title_5">Install Jupyter - Optional</h2>

        <p>Jupyter offers a really beautiful interface to play with Python. You can install Jupyter by following the instructions in the <a href="https://jupyter.org/install">Jupyter Website</a>.</p>

        <h2 id="title_6">Install Vertica ML Python</h2>
        <div><?php include('connection-include.php'); ?></div>
        <div><?php include('include/footer.php'); ?></div>
        </div>
      </div>
    </body>
</html>