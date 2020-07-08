<!DOCTYPE html>
<html>
    <?php include('include/head.php'); ?>
    <body>
      <div><?php include('include/header.php'); ?></div>
      <div id="content">
        <div><?php include('include/logo.php'); ?></div>
        <h1>Installation</h1>
        <p>
          Setting up the environment is crucial.
          <ul class="ul_content">
            <a href="#title_1"><li>Install Vertica</li></a>
            <a href="#title_2"><li>Making the VM IP address static - <b>Optional</b></li></a>
            <a href="#title_3"><li>Create a DataBase DSN (Data Source Name) - <b>Optional</b></li></a>
            <a href="#title_4"><li>Install Python3</li></a>
            <a href="#title_5"><li>Install Jupyter - <b>Optional</b></li></a>
            <a href="#title_6"><li>Install Vertica ML Python</li></a>
          </ul>
        </p>
        <p>If you already have Vertica and Python3 installed in your environment. Run the following command to install VerticaPy.</p><br>
        <div class="highlight"><pre><span></span>pip3 install vertica_ml_python</pre></div>
        <h2 id="title_1">Vertica Installation</h2>
        <p>Vertica is the most advanced analytics data warehouse based on a massively scalable architecture with the broadest set of analytical functions spanning event and time series, pattern matching, geospatial and end-to-end in-database machine learning. Vertica enables you to easily apply these powerful functions to the largest and most demanding analytical workloads, arming you and your customers with predictive business insights faster than any analytics data warehouse in the market.</p><br>
        <p>Vertica provides a unified analytics platform across major public clouds and on-premises data centers and integrates data in cloud object storage and HDFS without forcing you to move any of your data.</p><br>
        <p>To learn more about the Vertica Database, you can go to the <a href="https://www.vertica.com/about/">Vertica Official Website</a>.</p><br>
        <p>If you already have Vertica installed in your Machine, you can skip this step. Otherwise you have 3 options to get Vertica for free.</p>
        <p><ul class="ul_content">
            <li>If you have a Linux machine, you can install Vertica Community Edition. Please see <a href="https://www.youtube.com/watch?v=D5SbzVVR_Ps">this video</a>.</li>
            <li>If you don't have a Linux machine, you can use the Vertica Community Edition VM. In this case, follow the instructions of the <a href="https://www.vertica.com/docs/VMs/Vertica_CE_VM_Download_and_Startup_Instructions.pdf">Vertica Community Edition Virtual Machine Installation Guide</a>.</li>
            <li>You can also use a Vertica machine remotely for free by signing up for the <a href="https://academy.vertica.com/">Vertica Academy</a>. After joining, you can send an email to the <a href="https://academy.vertica.com/path-player?courseid=essentials9x&unit=5e41c0fdc1ba3177028b4584Unit">Vertica Academy team</a> to request a machine.</li>
          </ul>
        </p>

       <h2 id="title_2">Making the VM IP address static - Optional</h2>

        <p>This step is essential if you want to work on your own OS rather than directly in the VM. To allow your OS to communicate with the VM, open the parameters of your VM and create a second adapter (Adapter 2) and set it to bridged adapter.</p>
        <br>
        <center>
          <img src="img/network/0.png" width="30%">
          <img src="img/network/1.png" width="30%">
          <img src="img/network/2.png" width="30%">
        </center> 
        <br>
        <p>
          Look at the IP address of your machine using the <b>ifconfig</b> command in your terminal.
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
        <p>From now on, you simply need to turn on your VM and connect to it using SSH: dbadmin@192.168.1.150</p>

        <h2 id="title_3">Create a DataBase DSN (Data Source Name) - Optional</h2>

        <p>Creating DSN can save you a lot of time when working with databases. To create a DSN, you can follow the instructions in the <a href="https://www.vertica.com/docs/9.2.x/HTML/Content/Authoring/ConnectingToVertica/InstallingDrivers/CreatingAnODBCDataSourceNameDSN.htm">Vertica documentation</a>.</p><br>

        <p>DSN configuration could be a little bit more complicated in MacOS.</p><br>

        <p>Vertica includes its own drivers for different platforms, including MacOS. Just download the drivers from the <a href="https://my.vertica.com/download/vertica/client-drivers/">Vertica website</a>.</p><br>

        <p>Select your version of Vertica and download the corresponding package to install in your operating system. Vertica client drivers after and 8.1.0 offer backwards compatiblity. So, for example, a Vertica 8.1.1 database is compatible with version 9 client drivers.</p>
    
    <p>Install the package by double clicking in the file. Since the package was downloaded from an unknown developer, MacOS will ask for permission to install it. To finish the installation, go to System Preferences > Security and Privacy > General and accept the installation.</p><br>
        <p>By default, the drivers install to the directory "/Library/Vertica/ODBC/lib".</p>
        <br>
        <div class="highlight"><pre><span></span>root@ubuntu:~$ ls /Library/Vertica/ODBC/lib/</pre></div>
        <br>
        <p>There are two files:</p>
        <ul class="ul_content">
        <li><b>libverticaodbc.dylib</b> - This is the driver</li>
        <li><b>vertica.ini</b> - Vertica configuration file template for the driver</li>
        </ul>
        <p>After installing the drivers, we should modify some configuration files:</p>
        <ul class="ul_content">
          <li> /Library/Vertica/ODBC/lib/vertica.ini</li>
          <li>~/Library/ODBC/odbc.ini</li>
          <li> ~/Library/ODBC/odbcinst.ini</li>
          <li>VERTICAINI and ODBCINI system parameters</li>
        </ul>
        <p>The easiest way to do this is to use ODBC Administrator, a GUI Tool to configure the odbc.ini and odbcinst.ini
        The ODBC Administrator can be downloaded from the <a href="https://support.apple.com/kb/DL895?locale=en_US">Apple website</a>.</p>

        <p>After installing and running the ODBC Administrator, you can add a DSN:</p><br>

        <center>
          <img src="img/DSN/1.png" width="30%">
          <img src="img/DSN/2.png" width="30%">
          <img src="img/DSN/3.png" width="30%">
          <img src="img/DSN/4.png" width="30%">
          <img src="img/DSN/5.png" width="30%">
          <img src="img/DSN/6.png" width="30%">
        </center><br>

        <p>We confirm that the DSN was added by checking the contents of odbc.ini.</p><br>
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

        <p>The vertica.ini configuration file is preconfigured, but you can copy it to the same directory as the other ODBC files.</p><br>
        <div class="highlight"><pre><span></span>root@ubuntu:~$ cp /Library/Vertica/ODBC/lib/vertica.ini ~/Library/ODBC/

# Change the encoding parameter from UTF-32 to UTF-16 in the copied vertica.ini.
[Driver]
ErrorMessagesPath=/Library/Vertica/ODBC/messages/
ODBCInstLib=/usr/lib/libiodbcinst.dylib
DriverManagerEncoding=UTF-16</pre></div><br>

        <p>The final step is to add the environment variables VERTICAINI and ODBCINI. You can add these to your bash_profile (or /etc/profile for a system-wide change).</p><br>

        <div class="highlight"><pre><span></span>root@ubuntu:~$ vim ~/.bash_profile

## Parameters for ODBC-Vertica compatibility
[?]
export ODBCINI=/Library/ODBC/odbc.ini
export VERTICAINI=/Library/ODBC/vertica.ini
[?]</pre></div>

        <h2 id="title_4">Install Python3</h2>

        <p>Installing Python3 is as easy as downloading a file. Follow the instructions in the <a href="https://www.python.org/downloads/">Python website.</a></p>

        <h2 id="title_5">Install Jupyter - Optional</h2>

        <p>Jupyter offers a really beautiful interface interact with Python. You can install Jupyter by following the instructions in the <a href="https://jupyter.org/install">Jupyter website</a>.</p>

        <h2 id="title_6">Install Vertica ML Python</h2>
        <div><?php include('connection-include.php'); ?></div>
        <div><?php include('include/footer.php'); ?></div>
        </div>
      </div>
    </body>
</html>
