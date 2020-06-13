<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="new_auto_connection">new_auto_connection<a class="anchor-link" href="#new_auto_connection">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">new_auto_connection</span><span class="p">(</span><span class="n">dsn</span><span class="p">:</span> <span class="nb">dict</span><span class="p">,</span>
                    <span class="n">method</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">,</span>
                    <span class="n">name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;VML&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Saves a connection to automatically create DB cursors. It will create a file which will be used to automatically set up a connection when it is needed. It helps you to avoid redundant cursors creation.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
<tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
<tr> <td><div class="param_name">dsn</div></td> <td><div class="type">dict</div></td> <td><div class = "no">&#10060;</div></td> <td>Dictionnary containing the information to set up the connection.<br>
                                                    <ul>
                                                        <li><b>database :</b> Database Name</li>
                                                        <li><b>driver :</b> ODBC driver (only for pyodbc)</li>
                                                        <li><b>host :</b> Server ID</li>
                                                        <li><b>password :</b> User Password</li>
                                                        <li><b>port :</b> Database Port (optional, default: 5433)</li>
                                                        <li><b>user :</b> User ID (optional, default: dbadmin)</li>
                                                        </ul></td> </tr>
<tr> <td><div class="param_name">method</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Method used to save the connection.<br>
                                                    <ul>
                                                        <li><b>auto :</b> uses vertica_python if vertica_python installed, otherwise pyodbc, otherwise jaydebeapi.</li>
                                                        <li><b>pyodbc :</b> ODBC.</li>
                                                        <li><b>jaydebeapi :</b> JDBC.</li>
                                                        <li><b>vertica_python :</b> Vertica Python Native Client (recommended).</li>
                                                        </ul></td> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Name of the auto connection.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.connections.connect</span> <span class="k">import</span> <span class="o">*</span>
<span class="c1"># JDBC Example</span>
<span class="n">new_auto_connection</span><span class="p">({</span><span class="s2">&quot;host&quot;</span><span class="p">:</span> <span class="s2">&quot;10.211.55.14&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;port&quot;</span><span class="p">:</span> <span class="s2">&quot;5433&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;database&quot;</span><span class="p">:</span> <span class="s2">&quot;testdb&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;password&quot;</span><span class="p">:</span> <span class="s2">&quot;XxX&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;user&quot;</span><span class="p">:</span> <span class="s2">&quot;dbadmin&quot;</span><span class="p">},</span> 
                     <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;jaydebeapi&quot;</span><span class="p">,</span> 
                     <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;my_auto_connection&quot;</span><span class="p">)</span>
<span class="c1"># ODBC Example</span>
<span class="n">new_auto_connection</span><span class="p">({</span><span class="s2">&quot;host&quot;</span><span class="p">:</span> <span class="s2">&quot;10.211.55.14&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;port&quot;</span><span class="p">:</span> <span class="s2">&quot;5433&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;database&quot;</span><span class="p">:</span> <span class="s2">&quot;testdb&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;password&quot;</span><span class="p">:</span> <span class="s2">&quot;XxX&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;user&quot;</span><span class="p">:</span> <span class="s2">&quot;dbadmin&quot;</span><span class="p">,</span>
                     <span class="s2">&quot;driver&quot;</span><span class="p">:</span> <span class="s2">&quot;/Library/Vertica/ODBC/lib/libverticaodbc.dylib&quot;</span><span class="p">},</span> 
                     <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;pyodbc&quot;</span><span class="p">,</span> 
                     <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;my_auto_connection&quot;</span><span class="p">)</span>
<span class="c1"># Vertica Python Example</span>
<span class="n">new_auto_connection</span><span class="p">({</span><span class="s2">&quot;host&quot;</span><span class="p">:</span> <span class="s2">&quot;10.211.55.14&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;port&quot;</span><span class="p">:</span> <span class="s2">&quot;5433&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;database&quot;</span><span class="p">:</span> <span class="s2">&quot;testdb&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;password&quot;</span><span class="p">:</span> <span class="s2">&quot;XxX&quot;</span><span class="p">,</span> 
                     <span class="s2">&quot;user&quot;</span><span class="p">:</span> <span class="s2">&quot;dbadmin&quot;</span><span class="p">},</span> 
                     <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;vertica_python&quot;</span><span class="p">,</span> 
                     <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;my_auto_connection&quot;</span><span class="p">)</span>
<span class="c1"># Set the main auto connection</span>
<span class="n">change_auto_connection</span><span class="p">(</span><span class="s2">&quot;my_auto_connection&quot;</span><span class="p">)</span>

<span class="c1"># Displays the available connections</span>
<span class="n">available_auto_connection</span><span class="p">()</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The available connections are the following: my_auto_connection, VerticaDSN, VML, VML2
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt output_prompt">Out[5]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>[&#39;my_auto_connection&#39;, &#39;VerticaDSN&#39;, &#39;VML&#39;, &#39;VML2&#39;]</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="See-Also">See Also<a class="anchor-link" href="#See-Also">&#182;</a></h3><table id="seealso">
    <tr><td><a href="../change_auto_connection/index.php">change_auto_connection</a></td> <td> Changes the current auto creation.</td></tr>
    <tr><td><a href="../read_auto_connect/index.php">read_auto_connect</a></td> <td> Automatically creates a connection.</td></tr>
    <tr><td><a href="../vertica_cursor/index.php">vertica_cursor</a></td> <td> Creates a Vertica Database cursor using the input method.</td></tr>
</table>
</div>
</div>
</div>