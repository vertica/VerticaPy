<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="read_json">read_json<a class="anchor-link" href="#read_json">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">read_json</span><span class="p">(</span><span class="n">path</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
          <span class="n">cursor</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span> 
          <span class="n">schema</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;public&#39;</span><span class="p">,</span> 
          <span class="n">table_name</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s1">&#39;&#39;</span><span class="p">,</span>
          <span class="n">usecols</span><span class="p">:</span> <span class="nb">list</span> <span class="o">=</span> <span class="p">[],</span>
          <span class="n">new_name</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">=</span> <span class="p">{},</span>
          <span class="n">insert</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Ingests a JSON file using flex tables.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">path</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Absolute path where the JSON file is located.</td> </tr>
    <tr> <td><div class="param_name">cursor</div></td> <td><div class="type">DBcursor</div></td> <td><div class = "yes">&#10003;</div></td> <td>Vertica DB cursor.</td> </tr>
    <tr> <td><div class="param_name">schema</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Schema where the JSON file will be ingested.</td> </tr>
    <tr> <td><div class="param_name">table_name</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>Final relation name.</td> </tr>
    <tr> <td><div class="param_name">usecols</div></td> <td><div class="type">list</div></td> <td><div class = "yes">&#10003;</div></td> <td>List of the JSON parameters to ingest. The other ones will be ignored. If empty all the JSON parameters will be ingested.</td> </tr>
    <tr> <td><div class="param_name">new_name</div></td> <td><div class="type">dict</div></td> <td><div class = "yes">&#10003;</div></td> <td>Dictionary of the new columns name. If the JSON file is nested, it is advised to change the final names as special characters will be included. For example, {"param": {"age": 3, "name": Badr}, "date": 1993-03-11} will create 3 columns: "param.age", "param.name" and "date". You can rename these columns using the 'new_name' parameter with the following dictionary: {"param.age": "age", "param.name": "name"}</td> </tr>
    <tr> <td><div class="param_name">insert</div></td> <td><div class="type">bool</div></td> <td><div class = "yes">&#10003;</div></td> <td>If set to True, the data will be ingested to the input relation. The JSON parameters must be the same than the input relation otherwise they will not be ingested.</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>vDataFrame</b> : The vDataFrame of the relation.</p>

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
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.utilities</span> <span class="k">import</span> <span class="o">*</span>
<span class="c1"># Ingesting the JSON file</span>
<span class="n">read_json</span><span class="p">(</span><span class="s2">&quot;titanic.json&quot;</span><span class="p">,</span> 
          <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic_dataset&quot;</span><span class="p">,</span>
          <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The table &#34;public&#34;.&#34;titanic_dataset&#34; has been successfully created.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.embarked</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>record_timestamp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.passengerid</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.pclass</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.ticket</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>recordid</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>datasetid</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Collander, Mr. Erik Gustaf</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">343</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">13.0</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">248740</td><td style="border: 1px solid white;">835634b93c8f759537a89daa01c3c3658e934617</td><td style="border: 1px solid white;">28.0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Moen, Mr. Sigurd Hansen</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">76</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7.65</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">348123</td><td style="border: 1px solid white;">97941a419e5cf6a4bb65147a7a21d7025c8a6e1b</td><td style="border: 1px solid white;">25.0</td><td style="border: 1px solid white;">F G73</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Jensen, Mr. Hans Peder</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">641</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">7.8542</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">350050</td><td style="border: 1px solid white;">b762da1fa9f7f7765bc14006d9f5b8fc1d2d5177</td><td style="border: 1px solid white;">20.0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">Palsson, Mrs. Nils (Alma Cornelia Berglund)</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">568</td><td style="border: 1px solid white;">3</td><td style="border: 1px solid white;">21.075</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">349909</td><td style="border: 1px solid white;">dc455b086d203605705820911c0aaa98467bcd41</td><td style="border: 1px solid white;">29.0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Davidson, Mr. Thornton</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">672</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">52.0</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">F.C. 12750</td><td style="border: 1px solid white;">5aa00b39a93376656528f1c7d929a297e31e1a20</td><td style="border: 1px solid white;">31.0</td><td style="border: 1px solid white;">B71</td><td style="border: 1px solid white;">titanic-passengers</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[8]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic_dataset, Number of rows: 891, Number of columns: 15</pre>
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
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Ingesting the JSON file and renaming some columns</span>
<span class="n">read_json</span><span class="p">(</span><span class="s2">&quot;titanic.json&quot;</span><span class="p">,</span> 
          <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic_dataset&quot;</span><span class="p">,</span>
          <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
          <span class="n">new_name</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;fields.fare&quot;</span><span class="p">:</span> <span class="s2">&quot;fare&quot;</span><span class="p">,</span>
                      <span class="s2">&quot;fields.sex&quot;</span><span class="p">:</span> <span class="s2">&quot;sex&quot;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The table &#34;public&#34;.&#34;titanic_dataset&#34; has been successfully created.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.ticket</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>recordid</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.age</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.cabin</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>datasetid</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.parch</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.name</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.survived</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.embarked</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.sibsp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>record_timestamp</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.passengerid</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fields.pclass</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">248740</td><td style="border: 1px solid white;">835634b93c8f759537a89daa01c3c3658e934617</td><td style="border: 1px solid white;">28.0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">titanic-passengers</td><td style="border: 1px solid white;">13.0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Collander, Mr. Erik Gustaf</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">343</td><td style="border: 1px solid white;">2</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">348123</td><td style="border: 1px solid white;">97941a419e5cf6a4bb65147a7a21d7025c8a6e1b</td><td style="border: 1px solid white;">25.0</td><td style="border: 1px solid white;">F G73</td><td style="border: 1px solid white;">titanic-passengers</td><td style="border: 1px solid white;">7.65</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Moen, Mr. Sigurd Hansen</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">76</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">350050</td><td style="border: 1px solid white;">b762da1fa9f7f7765bc14006d9f5b8fc1d2d5177</td><td style="border: 1px solid white;">20.0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">titanic-passengers</td><td style="border: 1px solid white;">7.8542</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Jensen, Mr. Hans Peder</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">641</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">349909</td><td style="border: 1px solid white;">dc455b086d203605705820911c0aaa98467bcd41</td><td style="border: 1px solid white;">29.0</td><td style="border: 1px solid white;">None</td><td style="border: 1px solid white;">titanic-passengers</td><td style="border: 1px solid white;">21.075</td><td style="border: 1px solid white;">4</td><td style="border: 1px solid white;">Palsson, Mrs. Nils (Alma Cornelia Berglund)</td><td style="border: 1px solid white;">female</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">568</td><td style="border: 1px solid white;">3</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">F.C. 12750</td><td style="border: 1px solid white;">5aa00b39a93376656528f1c7d929a297e31e1a20</td><td style="border: 1px solid white;">31.0</td><td style="border: 1px solid white;">B71</td><td style="border: 1px solid white;">titanic-passengers</td><td style="border: 1px solid white;">52.0</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">Davidson, Mr. Thornton</td><td style="border: 1px solid white;">male</td><td style="border: 1px solid white;">False</td><td style="border: 1px solid white;">S</td><td style="border: 1px solid white;">1</td><td style="border: 1px solid white;">2016-09-20 15:34:51.313</td><td style="border: 1px solid white;">672</td><td style="border: 1px solid white;">1</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[9]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic_dataset, Number of rows: 891, Number of columns: 15</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Ingesting the JSON file, using some columns and</span>
<span class="c1"># renaming some columns</span>
<span class="n">read_json</span><span class="p">(</span><span class="s2">&quot;titanic.json&quot;</span><span class="p">,</span> 
          <span class="n">table_name</span> <span class="o">=</span> <span class="s2">&quot;titanic_dataset&quot;</span><span class="p">,</span>
          <span class="n">schema</span> <span class="o">=</span> <span class="s2">&quot;public&quot;</span><span class="p">,</span>
          <span class="n">usecols</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;fields.fare&quot;</span><span class="p">,</span> <span class="s2">&quot;fields.sex&quot;</span><span class="p">],</span>
          <span class="n">new_name</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;fields.fare&quot;</span><span class="p">:</span> <span class="s2">&quot;fare&quot;</span><span class="p">,</span>
                      <span class="s2">&quot;fields.sex&quot;</span><span class="p">:</span> <span class="s2">&quot;sex&quot;</span><span class="p">})</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>The table &#34;public&#34;.&#34;titanic_dataset&#34; has been successfully created.
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b></b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>fare</b></td><td style="font-size:1.02em;background-color:#263133;color:white"><b>sex</b></td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>0</b></td><td style="border: 1px solid white;">13.0</td><td style="border: 1px solid white;">male</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>1</b></td><td style="border: 1px solid white;">7.65</td><td style="border: 1px solid white;">male</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>2</b></td><td style="border: 1px solid white;">7.8542</td><td style="border: 1px solid white;">male</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>3</b></td><td style="border: 1px solid white;">21.075</td><td style="border: 1px solid white;">female</td></tr><tr ><td style="font-size:1.02em;background-color:#263133;color:white"><b>4</b></td><td style="border: 1px solid white;">52.0</td><td style="border: 1px solid white;">male</td></tr><tr><td style="border-top: 1px solid white;background-color:#263133;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: titanic_dataset, Number of rows: 891, Number of columns: 2</pre>
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
    <tr><td><a href="../read_csv/index.php">read_csv</a></td> <td>Ingests a CSV file in the Vertica DB.</td></tr>
</table>
</div>
</div>
</div>