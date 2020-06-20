<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="vDataFrame.join">vDataFrame.join<a class="anchor-link" href="#vDataFrame.join">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">input_relation</span><span class="p">,</span> 
                <span class="n">on</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">=</span> <span class="p">{},</span>
                <span class="n">how</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;natural&#39;</span><span class="p">,</span>
                <span class="n">expr1</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;*&#39;</span><span class="p">],</span>
                <span class="n">expr2</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;*&#39;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Joins the vDataFrame with another one or an input relation.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str / vDataFrame</div></td> <td><div class = "no">&#10060;</div></td> <td>Relation used to do the merging.</td> </tr>
    <tr> <td><div class="param_name">on</div></td> <td><div class="type">dict</div></td> <td><div class = "yes">&#10003;</div></td> <td>Dictionary of all the different keys. The dict must be similar to the following: {"relationA_key1": "relationB_key1" ..., "relationA_keyk": "relationB_keyk"} where relationA is the current vDataFrame and relationB is the input relation or the input vDataFrame.</td> </tr>
    <tr> <td><div class="param_name">how</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Join Type.<br>
                                                    <ul>
                                                        <li><b>left :</b> Left Join.</li>
                                                        <li><b>right :</b> Right Join.</li>
                                                        <li><b>cross :</b> Cross Join.</li>
                                                        <li><b>full :</b> Full Outer Join.</li>
                                                        <li><b>natural :</b> Natural Join.</li>
                                                        <li><b>inner :</b> Inner Join. </li>
                                                        </ul></td> </tr>
    <tr> <td><div class="param_name">expr1</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the different columns to select from the current vDataFrame. Pure SQL must be written. Aliases can also be given. 'column' or 'column AS my_new_alias' are correct. Aliases are recommended to keep the track of the different features and not have ambiguous names. </td> </tr>
    <tr> <td><div class="param_name">expr2</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the different columns to select from the current vDataFrame. Pure SQL must be written. Aliases can also be given. 'column' or 'column AS my_new_alias' are correct. Aliases are recommended to keep the track of the different features and not have ambiguous names. </td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : object result of the join.</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">flights</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;public.flights&quot;</span><span class="p">)</span>
<span class="n">airports</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;public.airports&quot;</span><span class="p">)</span>
<span class="n">airlines</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;public.airlines&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">flights</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">airports</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">airlines</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>arrival_delay</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">-9</td><td style="border: 1px solid white;">11433</td><td style="border: 1px solid white;">2015-10-01 10:09:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-01 10:27:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-14</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">-4</td><td style="border: 1px solid white;">13930</td><td style="border: 1px solid white;">2015-10-01 13:57:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">6</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">11433</td><td style="border: 1px solid white;">2015-10-01 14:02:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-8</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-01 14:44:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-1</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: flights, Number of rows: 4068736, Number of columns: 6
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>COUNTRY</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>IATA_CODE</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>AIRPORT</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>LATITUDE</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>CITY</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>STATE</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>LONGITUDE</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">Lehigh Valley International Airport</td><td style="border: 1px solid white;">40.652360</td><td style="border: 1px solid white;">Allentown</td><td style="border: 1px solid white;">PA</td><td style="border: 1px solid white;">-75.440400</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABI</td><td style="border: 1px solid white;">Abilene Regional Airport</td><td style="border: 1px solid white;">32.411320</td><td style="border: 1px solid white;">Abilene</td><td style="border: 1px solid white;">TX</td><td style="border: 1px solid white;">-99.681900</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABQ</td><td style="border: 1px solid white;">Albuquerque International Sunport</td><td style="border: 1px solid white;">35.040220</td><td style="border: 1px solid white;">Albuquerque</td><td style="border: 1px solid white;">NM</td><td style="border: 1px solid white;">-106.609190</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABR</td><td style="border: 1px solid white;">Aberdeen Regional Airport</td><td style="border: 1px solid white;">45.449060</td><td style="border: 1px solid white;">Aberdeen</td><td style="border: 1px solid white;">SD</td><td style="border: 1px solid white;">-98.421830</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABY</td><td style="border: 1px solid white;">Southwest Georgia Regional Airport</td><td style="border: 1px solid white;">31.535520</td><td style="border: 1px solid white;">Albany</td><td style="border: 1px solid white;">GA</td><td style="border: 1px solid white;">-84.194470</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: airports, Number of rows: 322, Number of columns: 7
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>AIRLINE</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>IATA_CODE</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">American Airlines Inc.</td><td style="border: 1px solid white;">AA</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">Alaska Airlines Inc.</td><td style="border: 1px solid white;">AS</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">JetBlue Airways</td><td style="border: 1px solid white;">B6</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">Delta Air Lines Inc.</td><td style="border: 1px solid white;">DL</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td><td style="border: 1px solid white;">EV</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: airlines, Number of rows: 14, Number of columns: 2
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Cross Join</span>
<span class="n">airports</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">airports</span><span class="p">,</span> 
              <span class="n">how</span> <span class="o">=</span> <span class="s2">&quot;cross&quot;</span><span class="p">,</span> 
              <span class="n">expr1</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;IATA_CODE AS airport1&quot;</span><span class="p">,</span> 
                       <span class="s2">&quot;LATITUDE AS airport1_latitude&quot;</span><span class="p">,</span> 
                       <span class="s2">&quot;LONGITUDE AS airport1_longitude&quot;</span><span class="p">],</span>
              <span class="n">expr2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;IATA_CODE AS airport2&quot;</span><span class="p">,</span> 
                       <span class="s2">&quot;LATITUDE AS airport2_latitude&quot;</span><span class="p">,</span> 
                       <span class="s2">&quot;LONGITUDE AS airport2_longitude&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>airport1</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>airport1_latitude</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>airport1_longitude</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>airport2</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>airport2_latitude</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>airport2_longitude</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.652360</td><td style="border: 1px solid white;">-75.440400</td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.652360</td><td style="border: 1px solid white;">-75.440400</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.652360</td><td style="border: 1px solid white;">-75.440400</td><td style="border: 1px solid white;">ABI</td><td style="border: 1px solid white;">32.411320</td><td style="border: 1px solid white;">-99.681900</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.652360</td><td style="border: 1px solid white;">-75.440400</td><td style="border: 1px solid white;">ABQ</td><td style="border: 1px solid white;">35.040220</td><td style="border: 1px solid white;">-106.609190</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.652360</td><td style="border: 1px solid white;">-75.440400</td><td style="border: 1px solid white;">ABR</td><td style="border: 1px solid white;">45.449060</td><td style="border: 1px solid white;">-98.421830</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.652360</td><td style="border: 1px solid white;">-75.440400</td><td style="border: 1px solid white;">ABY</td><td style="border: 1px solid white;">31.535520</td><td style="border: 1px solid white;">-84.194470</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[14]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: join, Number of rows: 103684, Number of columns: 6</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Left Join</span>
<span class="n">flights</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">airlines</span><span class="p">,</span>
             <span class="n">how</span> <span class="o">=</span> <span class="s2">&quot;left&quot;</span><span class="p">,</span>
             <span class="n">on</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;airline&quot;</span><span class="p">:</span> <span class="s2">&quot;IATA_CODE&quot;</span><span class="p">},</span>
             <span class="n">expr1</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;*&quot;</span><span class="p">],</span>
             <span class="n">expr2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;AIRLINE AS airline_long&quot;</span><span class="p">])</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>arrival_delay</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>airline_long</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">2015-08-16 20:12:00</td><td style="border: 1px solid white;">14</td><td style="border: 1px solid white;">5</td><td style="border: 1px solid white;">DTW</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">2015-08-17 10:07:00</td><td style="border: 1px solid white;">29</td><td style="border: 1px solid white;">27</td><td style="border: 1px solid white;">DTW</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">2015-08-17 10:25:00</td><td style="border: 1px solid white;">19</td><td style="border: 1px solid white;">10</td><td style="border: 1px solid white;">ATL</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">2015-08-17 14:00:00</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">61</td><td style="border: 1px solid white;">ORD</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">2015-08-17 14:12:00</td><td style="border: 1px solid white;">-5</td><td style="border: 1px solid white;">-17</td><td style="border: 1px solid white;">DTW</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[16]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: join, Number of rows: 4068736, Number of columns: 7</pre>
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
    <tr><td><a href="../append/index.php">vDataFrame.append</a></td> <td>Merges the vDataFrame with another relation.</td></tr>
    <tr><td><a href="../groupby/index.php">vDataFrame.groupby</a></td> <td>Aggregates the vDataFrame.</td></tr>
    <tr><td><a href="../sort/index.php">vDataFrame.sort</a></td> <td>Sorts the vDataFrame.</td></tr>
</table>
</div>
</div>
</div>