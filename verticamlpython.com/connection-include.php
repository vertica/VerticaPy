<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>To install Vertica ML Python, just run the following <b>pip</b> command.</p>
<div class="highlight"><pre><span></span>root@ubuntu:~$ pip3 install vertica_ml_python
</pre></div>
<p>To be able to connect to Vertica, you'll need to install one of the following modules.</p>
<ul class="ul_content">
<li><b>vertica_python</b> (Native Python Client)</li>
<li><b>pyodbc</b> (ODBC) </li>
<li><b>jaydebeapi</b> (JDBC)</li>
</ul>
<p>These modules will give you the possibility to create DataBase cursor which will be used to communicate with your Vertica DataBase.</p>
<p>For example, use the following command to install the <b>vertica_python</b> module.</p>
<div class="highlight"><pre><span></span>root@ubuntu:~$ pip3 install vertica_python
</pre></div>
<p>If you have created a DSN, you can easily set-up a connection using the following command.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#</span>
<span class="c1">#</span>
<span class="c1"># vertica_cursor</span>
<span class="c1">#</span>
<span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="n">vertica_cursor</span>
<span class="n">cur</span> <span class="o">=</span> <span class="n">vertica_cursor</span><span class="p">(</span><span class="s2">&quot;VerticaDSN&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The Vertica Native Python Client <b>vertica_python</b> is the easiest one to set-up.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#</span>
<span class="c1">#</span>
<span class="c1"># vertica_python</span>
<span class="c1">#</span>
<span class="kn">import</span> <span class="nn">vertica_python</span>

<span class="c1"># Connection using all the DSN information</span>
<span class="n">conn_info</span> <span class="o">=</span> <span class="p">{</span><span class="s1">&#39;host&#39;</span><span class="p">:</span> <span class="s2">&quot;10.211.55.14&quot;</span><span class="p">,</span> 
             <span class="s1">&#39;port&#39;</span><span class="p">:</span> <span class="mi">5433</span><span class="p">,</span> 
             <span class="s1">&#39;user&#39;</span><span class="p">:</span> <span class="s2">&quot;dbadmin&quot;</span><span class="p">,</span> 
             <span class="s1">&#39;password&#39;</span><span class="p">:</span> <span class="s2">&quot;XxX&quot;</span><span class="p">,</span> 
             <span class="s1">&#39;database&#39;</span><span class="p">:</span> <span class="s2">&quot;testdb&quot;</span><span class="p">}</span>
<span class="n">cur</span> <span class="o">=</span> <span class="n">vertica_python</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="o">**</span> <span class="n">conn_info</span><span class="p">)</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>

<span class="c1"># Connection using directly the DSN</span>
<span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="n">to_vertica_python_format</span> 
<span class="n">dsn</span> <span class="o">=</span> <span class="s2">&quot;VerticaDSN&quot;</span>
<span class="n">cur</span> <span class="o">=</span> <span class="n">vertica_python</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="o">**</span> <span class="n">to_vertica_python_format</span><span class="p">(</span><span class="n">dsn</span><span class="p">),</span> <span class="n">autocommit</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>To create an ODBC connection, <b>pyodbc</b> offers you two possibilities (one with the DSN and one with all the credentials).</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#</span>
<span class="c1">#</span>
<span class="c1"># pyodbc</span>
<span class="c1">#</span>
<span class="kn">import</span> <span class="nn">pyodbc</span>

<span class="c1"># Connection using all the DSN information</span>
<span class="n">driver</span> <span class="o">=</span> <span class="s2">&quot;/Library/Vertica/ODBC/lib/libverticaodbc.dylib&quot;</span>
<span class="n">server</span> <span class="o">=</span> <span class="s2">&quot;10.211.55.14&quot;</span>
<span class="n">database</span> <span class="o">=</span> <span class="s2">&quot;testdb&quot;</span>
<span class="n">port</span> <span class="o">=</span> <span class="s2">&quot;5433&quot;</span>
<span class="n">uid</span> <span class="o">=</span> <span class="s2">&quot;dbadmin&quot;</span>
<span class="n">pwd</span> <span class="o">=</span> <span class="s2">&quot;XxX&quot;</span>
<span class="n">dsn</span> <span class="o">=</span> <span class="p">(</span><span class="s2">&quot;DRIVER=</span><span class="si">{}</span><span class="s2">; SERVER=</span><span class="si">{}</span><span class="s2">; DATABASE=</span><span class="si">{}</span><span class="s2">; PORT=</span><span class="si">{}</span><span class="s2">; UID=</span><span class="si">{}</span><span class="s2">; PWD=</span><span class="si">{}</span><span class="s2">;&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">driver</span><span class="p">,</span> 
                                                                             <span class="n">server</span><span class="p">,</span> 
                                                                             <span class="n">database</span><span class="p">,</span> 
                                                                             <span class="n">port</span><span class="p">,</span> 
                                                                             <span class="n">uid</span><span class="p">,</span> 
                                                                             <span class="n">pwd</span><span class="p">)</span>
<span class="n">cur</span> <span class="o">=</span> <span class="n">pyodbc</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="n">dsn</span><span class="p">)</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>

<span class="c1"># Connection using directly the DSN</span>
<span class="n">dsn</span> <span class="o">=</span> <span class="p">(</span><span class="s2">&quot;DSN=VerticaDSN&quot;</span><span class="p">)</span>
<span class="n">cur</span> <span class="o">=</span> <span class="n">pyodbc</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="n">dsn</span><span class="p">,</span> <span class="n">autocommit</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The <b>jaydebeapi</b> module offers you the possibility to set-up a JDBC connection.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#</span>
<span class="c1">#</span>
<span class="c1"># jaydebeapi</span>
<span class="c1">#</span>
<span class="kn">import</span> <span class="nn">jaydebeapi</span>

<span class="c1"># Vertica Server Details</span>
<span class="n">database</span> <span class="o">=</span> <span class="s2">&quot;testdb&quot;</span>
<span class="n">hostname</span> <span class="o">=</span> <span class="s2">&quot;10.211.55.14&quot;</span>
<span class="n">port</span> <span class="o">=</span> <span class="s2">&quot;5433&quot;</span>
<span class="n">uid</span> <span class="o">=</span> <span class="s2">&quot;dbadmin&quot;</span>
<span class="n">pwd</span> <span class="o">=</span> <span class="s2">&quot;XxX&quot;</span>

<span class="c1"># Vertica JDBC class name</span>
<span class="n">jdbc_driver_name</span> <span class="o">=</span> <span class="s2">&quot;com.vertica.jdbc.Driver&quot;</span>

<span class="c1"># Vertica JDBC driver path</span>
<span class="n">jdbc_driver_loc</span> <span class="o">=</span> <span class="s2">&quot;/Library/Vertica/JDBC/vertica-jdbc-9.3.1-0.jar&quot;</span>

<span class="c1"># JDBC connection string</span>
<span class="n">connection_string</span> <span class="o">=</span> <span class="s1">&#39;jdbc:vertica://&#39;</span> <span class="o">+</span> <span class="n">hostname</span> <span class="o">+</span> <span class="s1">&#39;:&#39;</span> <span class="o">+</span> <span class="n">port</span> <span class="o">+</span> <span class="s1">&#39;/&#39;</span> <span class="o">+</span> <span class="n">database</span>
<span class="n">url</span> <span class="o">=</span> <span class="s1">&#39;</span><span class="si">{}</span><span class="s1">:user=</span><span class="si">{}</span><span class="s1">;password=</span><span class="si">{}</span><span class="s1">&#39;</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">connection_string</span><span class="p">,</span> <span class="n">uid</span><span class="p">,</span> <span class="n">pwd</span><span class="p">)</span>
<span class="n">conn</span> <span class="o">=</span> <span class="n">jaydebeapi</span><span class="o">.</span><span class="n">connect</span><span class="p">(</span><span class="n">jdbc_driver_name</span><span class="p">,</span> 
                          <span class="n">connection_string</span><span class="p">,</span> 
                          <span class="p">{</span><span class="s1">&#39;user&#39;</span><span class="p">:</span> <span class="n">uid</span><span class="p">,</span> <span class="s1">&#39;password&#39;</span><span class="p">:</span> <span class="n">pwd</span><span class="p">},</span> 
                          <span class="n">jars</span> <span class="o">=</span> <span class="n">jdbc_driver_loc</span><span class="p">)</span>
<span class="n">cur</span> <span class="o">=</span> <span class="n">conn</span><span class="o">.</span><span class="n">cursor</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>You can also, save your credentials in Vertica ML Python to avoid redundant cursors creations.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.connections.connect</span> <span class="k">import</span> <span class="o">*</span>
<span class="c1"># Save a new connection</span>
<span class="n">new_auto_connection</span><span class="p">({</span><span class="s2">&quot;host&quot;</span><span class="p">:</span> <span class="s2">&quot;10.211.55.14&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;port&quot;</span><span class="p">:</span> <span class="s2">&quot;5433&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;database&quot;</span><span class="p">:</span> <span class="s2">&quot;testdb&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;password&quot;</span><span class="p">:</span> <span class="s2">&quot;XxX&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;user&quot;</span><span class="p">:</span> <span class="s2">&quot;dbadmin&quot;</span><span class="p">},</span>
                    <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;vertica_python&quot;</span><span class="p">,</span> 
                    <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;VerticaDSN&quot;</span><span class="p">)</span>
<span class="c1"># Set the main auto connection</span>
<span class="n">change_auto_connection</span><span class="p">(</span><span class="s2">&quot;VerticaDSN&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
