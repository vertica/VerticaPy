<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.regexp">vDataFrame.regexp<a class="anchor-link" href="#vDataFrame.regexp">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">regexp</span><span class="p">(</span><span class="n">column</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                  <span class="n">pattern</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                  <span class="n">method</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;substr&quot;</span><span class="p">,</span> 
                  <span class="n">position</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span>
                  <span class="n">occurrence</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">1</span><span class="p">,</span>
                  <span class="n">replacement</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;&#39;</span><span class="p">,</span>
                  <span class="n">return_position</span> <span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span><span class="p">,</span>
                  <span class="n">name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Computes a new vcolumn based on regular expressions.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">column</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Input vcolumn used to compute the regular expression.</td> </tr>
    <tr> <td><div class="param_name">pattern</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>The regular expression.</td> </tr>
    <tr> <td><div class="param_name">method</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Method used to compute the regular expressions.<br>
                                                    <ul>
                                                        <li><b>count :</b> Returns the number times a regular expression matches each element of the input vcolumn. </li>
                                                        <li><b>ilike :</b> Returns True if the vcolumn element contains a match for the regular expression.</li>
                                                        <li><b>instr :</b> Returns the starting or ending position in a vcolumn element where a regular expression matches.</li>
                                                        <li><b>like :</b> Returns True if the vcolumn element matches the regular expression.</li>
                                                        <li><b>not_ilike :</b> Returns True if the vcolumn element does not match the case-insensitive regular expression.</li>
                                                        <li><b>not_like :</b> Returns True if the vcolumn element does not contain a match for the regular expression.</li>
                                                        <li><b>replace :</b> Replaces all occurrences of a substring that match a regular expression with another substring.</li>
                                                        <li><b>substr :</b> Returns the substring that matches a regular expression within a vcolumn.</li></ul></td> </tr>
    <tr> <td><div class="param_name">position</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>The number of characters from the start of the string where the function should start searching for matches.</td> </tr>
    <tr> <td><div class="param_name">occurrence</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Controls which occurrence of a pattern match in the string to return.</td> </tr>
    <tr> <td><div class="param_name">replacement</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>The string to replace matched substrings.</td> </tr>
    <tr> <td><div class="param_name">return_position</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Sets the position within the string to return.</td> </tr>
    <tr> <td><div class="param_name">name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>New feature name. If empty, a name will be generated.</td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : self</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">filmtv_movies</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;filmtv_movies&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">filmtv_movies</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">1</span><span class="p">))</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>avg_vote</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>description</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>genre</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>director</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>votes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>year</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>notes</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>title</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>filmtv_id</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>duration</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>country</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Mickey Rourke, Steve Guttenberg, Ellen Barkin, Daniel Stern, Kevin Bacon, Timothy Daly, Paul Reiser, Kelle Kipp, Colette Blonigan</td><td style="border: 1px solid white;">7.20</td><td style="border: 1px solid white;">Five boys from Baltimore are in the habit of meeting periodically for dinner and playing tricks together. One of them, Boogie, is in trouble because of his passion for betting: he owes a gangster two thousand dollars and to find them he keeps betting and losing. The others have marital or sex-related problems, and everyone has a few jokes.</td><td style="border: 1px solid white;">Comedy</td><td style="border: 1px solid white;">Barry Levinson</td><td style="border: 1px solid white;">15</td><td style="border: 1px solid white;">1982.00</td><td style="border: 1px solid white;">A cast of will be famous for Levinson's directorial debut. Very bitter and very well written: jokes are seen as a manifestation of immaturity rather than carefree; Rourke enhances his charge of beautiful darkness. Guttenberg is the good guy in the group.</td><td style="border: 1px solid white;">Diner</td><td style="border: 1px solid white;">18</td><td style="border: 1px solid white;">95</td><td style="border: 1px solid white;">United States</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: filmtv_movies, Number of rows: 53397, Number of columns: 12
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Retrieving the second actor</span>
<span class="n">filmtv_movies</span><span class="o">.</span><span class="n">regexp</span><span class="p">(</span><span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;actors&quot;</span><span class="p">,</span> 
                     <span class="n">pattern</span> <span class="o">=</span> <span class="s2">&quot;[^,]+&quot;</span><span class="p">,</span> 
                     <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;substr&quot;</span><span class="p">,</span>
                     <span class="n">occurrence</span> <span class="o">=</span> <span class="mi">2</span><span class="p">,</span>
                     <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;actor2&quot;</span><span class="p">)</span><span class="o">.</span><span class="n">select</span><span class="p">([</span><span class="s2">&quot;actors&quot;</span><span class="p">,</span> 
                                              <span class="s2">&quot;actor2&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actor2</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Mickey Rourke, Steve Guttenberg, Ellen Barkin, Daniel Stern, Kevin Bacon, Timothy Daly, Paul Reiser, Kelle Kipp, Colette Blonigan</td><td style="border: 1px solid white;"> Steve Guttenberg</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Gian Maria Volonté, Irene Papas, Gabriele Ferzetti, Salvo Randone, Laura Nucci, Mario Scaccia, Luigi Pistilli, Leopoldo Trieste</td><td style="border: 1px solid white;"> Irene Papas</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Alan Steel, Mary Arden, Sergio Ciani, Ivano Davoli, Giovanna Galletti, Aïché Nana, Charlie Charun, Gilberto Mazzi</td><td style="border: 1px solid white;"> Mary Arden</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">George Hilton, Ennio Girolami, Marta Padovan, Alfonso De La Vega, Venancio Muro, Alfonso Rojas, Luis Marin</td><td style="border: 1px solid white;"> Ennio Girolami</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Patsy Kensit, Stéphane Freiss, Mouss Diouf, Anne-Marie Pisani, Joseph Momo, Jean-Marc Truong, An Luu</td><td style="border: 1px solid white;"> Stéphane Freiss</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: filmtv_movies, Number of rows: 53397, Number of columns: 2</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Computing the Number of actors</span>
<span class="n">filmtv_movies</span><span class="o">.</span><span class="n">regexp</span><span class="p">(</span><span class="n">column</span> <span class="o">=</span> <span class="s2">&quot;actors&quot;</span><span class="p">,</span> 
                     <span class="n">pattern</span> <span class="o">=</span> <span class="s2">&quot;,&quot;</span><span class="p">,</span> 
                     <span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;count&quot;</span><span class="p">,</span>
                     <span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;nb_actors&quot;</span><span class="p">)</span>
<span class="n">filmtv_movies</span><span class="p">[</span><span class="s2">&quot;nb_actors&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">add</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
<span class="n">filmtv_movies</span><span class="o">.</span><span class="n">select</span><span class="p">([</span><span class="s2">&quot;actors&quot;</span><span class="p">,</span> <span class="s2">&quot;nb_actors&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>actors</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>nb_actors</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">Mickey Rourke, Steve Guttenberg, Ellen Barkin, Daniel Stern, Kevin Bacon, Timothy Daly, Paul Reiser, Kelle Kipp, Colette Blonigan</td><td style="border: 1px solid white;">9</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Gian Maria Volonté, Irene Papas, Gabriele Ferzetti, Salvo Randone, Laura Nucci, Mario Scaccia, Luigi Pistilli, Leopoldo Trieste</td><td style="border: 1px solid white;">8</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">Alan Steel, Mary Arden, Sergio Ciani, Ivano Davoli, Giovanna Galletti, Aïché Nana, Charlie Charun, Gilberto Mazzi</td><td style="border: 1px solid white;">8</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">George Hilton, Ennio Girolami, Marta Padovan, Alfonso De La Vega, Venancio Muro, Alfonso Rojas, Luis Marin</td><td style="border: 1px solid white;">7</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Patsy Kensit, Stéphane Freiss, Mouss Diouf, Anne-Marie Pisani, Joseph Momo, Jean-Marc Truong, An Luu</td><td style="border: 1px solid white;">7</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[11]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: filmtv_movies, Number of rows: 53397, Number of columns: 2</pre>
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
    <tr><td><a href="../eval/index.php">vDataFrame.eval</a></td> <td>Evaluates a customized expression.</td></tr>
</table>
</div>
</div>
</div>