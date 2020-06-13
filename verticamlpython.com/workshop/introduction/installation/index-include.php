<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Welcome">Welcome<a class="anchor-link" href="#Welcome">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<center>
  <img src="../../../img/Badr.jpg" width="30%" style="min-width: 300px;">
</center><p>Welcome to the Vertica ML Python lessons web pages. My name is Badr Ouali and I will try to guide you through the Data Science life cycle. I will introduce you the different functionalities of the Vertica ML Python library.</p>
<p>During the different lessons, we will also work on the different notions important to know to fulfill the Data Science journey. All the lessons will never take you more than 10 minutes (except the Vertica Installation and Configuration in case you do not have access to a Vertica cluster) so try to read all the content to not miss an important part.</p>
<p>To enjoy these lessons, you must first set-up the environment.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Setting-Up-the-environment">Setting Up the environment<a class="anchor-link" href="#Setting-Up-the-environment">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>To start playing with the Vertica ML Python Module, you must:</p>
<ul class="ul_content">
<li>Have access to a Machine where Vertica is installed</li>
<li>Install Python in your Machine</li>
<li>Install Vertica ML Python</li>
</ul>
<p>When everything is ready, you must connect to the Vertica DataBase. Different types of cursors are available:</p>
<ul class="ul_content">
<li>Native (with vertica_python)</li>
<li>ODBC (with pyodbc)</li>
<li>JDBC (with jaydebeapi)</li>
</ul>
<p>All the steps are explained in the <a href="../../../installation.php">Installation Page</a>.</p>
<h1 id="First-Steps">First Steps<a class="anchor-link" href="#First-Steps">&#182;</a></h1><p>Let's create for example a connection using vertica_python.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">vertica_python</span>

<span class="c1"># Connection using all the DSN information</span>
<span class="n">conn_info</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;host&#39;</span><span class="p">:</span> <span class="s2">&quot;10.211.55.14&quot;</span><span class="p">,</span> 
             <span class="s1">&#39;port&#39;</span><span class="p">:</span> <span class="mi">5433</span><span class="p">,</span> 
             <span class="s1">&#39;user&#39;</span><span class="p">:</span> <span class="s2">&quot;dbadmin&quot;</span><span class="p">,</span> 
             <span class="s1">&#39;password&#39;</span><span class="p">:</span> <span class="s2">&quot;XxX&quot;</span><span class="p">,</span> 
             <span class="s1">&#39;database&#39;</span><span class="p">:</span> <span class="s2">&quot;testdb&quot;</span><span class="p">}</span>
<span class="n">cur</span> <span class="o">=</span> <span class="n">vertica_python</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="o">**</span> <span class="n">conn_info</span><span class="p">)</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Most of the functions will use a DB cursor to query the DataBase. That's why it is possible to save a connection in the Vertica ML Python folder.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.connections.connect</span> <span class="k">import</span> <span class="o">*</span>
<span class="c1"># Save a new connection</span>
<span class="n">new_auto_connection</span><span class="p">({</span><span class="s2">&quot;host&quot;</span><span class="p">:</span> <span class="s2">&quot;10.211.55.14&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;port&quot;</span><span class="p">:</span> <span class="s2">&quot;5433&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;database&quot;</span><span class="p">:</span> <span class="s2">&quot;testdb&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;password&quot;</span><span class="p">:</span> <span class="s2">&quot;XxX&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;user&quot;</span><span class="p">:</span> <span class="s2">&quot;dbadmin&quot;</span><span class="p">},</span>
                    <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;jaydebeapi&quot;</span><span class="p">,</span> 
                    <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;VerticaDSN&quot;</span><span class="p">)</span>
<span class="c1"># Set the main auto connection</span>
<span class="n">change_auto_connection</span><span class="p">(</span><span class="s2">&quot;VerticaDSN&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>When no cursor is defined, Vertica ML Python will automatically create a new one using the saved credentials. Be Careful when using auto-connection as some connections may not be closed which can lead to an increase of the number of user sesssions (high concurrency). It can be used when the Vertica Cluster do not have many users.</p>
<p>As you noticed in the previous code, some parameters may have a specific format. The help module will summarize all the different parameters without going through the entire documentation.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">help</span><span class="p">(</span><span class="n">new_auto_connection</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Help on function new_auto_connection in module vertica_ml_python.connections.connect:

new_auto_connection(dsn:dict, method:str=&#39;auto&#39;, name:str=&#39;VML&#39;)
    ---------------------------------------------------------------------------
    Saves a connection to automatically create DB cursors. It will create a 
    file which will be used to automatically set up a connection when 
    it is needed. It helps you to avoid redundant cursors creation.
    
    Parameters
    ----------
    dsn: dict
            Dictionnary containing the information to set up the connection.
                    database : Database Name
                    driver   : ODBC driver (only for pyodbc)
                    host     : Server ID
                    password : User Password
                    port     : Database Port (optional, default: 5433)
                    user     : User ID (optional, default: dbadmin)
    method: str, optional
            Method used to save the connection.
            auto           : uses vertica_python if vertica_python installed, 
                    otherwise pyodbc, otherwise jaydebeapi.
            pyodbc         : ODBC.
            jaydebeapi     : JDBC.
            vertica_python : Vertica Python Native Client (recommended).
    name: str, optional
            Name of the auto connection.
    
    See Also
    --------
    change_auto_connection : Changes the current auto creation.
    read_auto_connect      : Automatically creates a connection.
    vertica_cursor         : Creates a Vertica Database cursor using the input method.

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>You have also access to the vHelp function which can help you to find the answers to numerous questions.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="n">vHelp</span>
<span class="n">vHelp</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_markdown rendered_html output_subarea ">
<center><img src='https://raw.githubusercontent.com/vertica/Vertica-ML-Python/master/img/logo.png' width="180px"></center><p>&#128226; Welcome to the <b>VERTICA ML PYTHON</b> help Module. You are about to use a new fantastic way to analyze your data !</p>
<p>You can learn quickly how to set up a connection, how to create a Virtual DataFrame and much more.</p>
<p>What do you want to know?</p>
<ul>
<li><b>[Enter  0]</b> Do you want to know why you should use this library ?</li>
<li><b>[Enter  1]</b> Do you want to know how to connect to your Vertica Database using Python and to Create a Virtual DataFrame ?</li>
<li><b>[Enter  2]</b> Do you want to know if your Vertica Version is compatible with the API ?</li>
<li><b>[Enter  3]</b> You don't have data to play with and you want to load an available dataset ?</li>
<li><b>[Enter  4]</b> Do you want to know other modules which can make your Data Science experience more complete ?</li>
<li><b>[Enter  5]</b> Do you want to look at a quick example ?</li>
<li><b>[Enter  6]</b> Do you want to look at the different functions available ?</li>
<li><b>[Enter  7]</b> Do you want to get a link to the VERTICA ML PYTHON wiki ?</li>
<li><b>[Enter  8]</b> Do you want to know how to display the Virtual DataFrame SQL code generation and the time elapsed to run the query ?</li>
<li><b>[Enter  9]</b> Do you want to know how to load your own dataset inside Vertica ?</li>
<li><b>[Enter 10]</b> Do you want to know how you writing direct SQL queries in Jupyter ?</li>
<li><b>[Enter 11]</b> Do you want to know how you could read and write using specific schemas ?</li>
<li><b>[Enter -1]</b> Exit</li>
</ul>

</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>-1
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_markdown rendered_html output_subarea ">
<p>Thank you for using the VERTICA ML PYTHON help.</p>

</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>You can now start playing with the Library functionalities !</p>

</div>
</div>
</div>