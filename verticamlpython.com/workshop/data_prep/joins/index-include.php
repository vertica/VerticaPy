<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Joins">Joins<a class="anchor-link" href="#Joins">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>When we explored our different datasets. We need to merge the different data sources. To be able to do it we need keys to join the data. To understand how to join data in Vertica ML Python, let's look at an example.</p>
<p>Let's use the US Flights 2015 datasets. 3 datasets are available.
We have information on the different flights.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[120]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python</span> <span class="k">import</span> <span class="o">*</span>
<span class="n">flights</span>  <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;flights&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">flights</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">-9</td><td style="border: 1px solid white;">11433</td><td style="border: 1px solid white;">2015-10-01 10:09:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-2</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-01 10:27:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-14</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">-4</td><td style="border: 1px solid white;">13930</td><td style="border: 1px solid white;">2015-10-01 13:57:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">6</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">11433</td><td style="border: 1px solid white;">2015-10-01 14:02:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-8</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">2015-10-01 14:44:00</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">-1</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: flights, Number of rows: 4068736, Number of columns: 6
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
<p>A dataset on the airports information is also available.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[106]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">airports</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;airports&quot;</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">airports</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>AIRPORT</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>LATITUDE</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>CITY</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>STATE</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>LONGITUDE</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>COUNTRY</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>IATA_CODE</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">Lehigh Valley International Airport</td><td style="border: 1px solid white;">40.65236</td><td style="border: 1px solid white;">Allentown</td><td style="border: 1px solid white;">PA</td><td style="border: 1px solid white;">-75.4404</td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABE</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">Abilene Regional Airport</td><td style="border: 1px solid white;">32.41132</td><td style="border: 1px solid white;">Abilene</td><td style="border: 1px solid white;">TX</td><td style="border: 1px solid white;">-99.6819</td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABI</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">Albuquerque International Sunport</td><td style="border: 1px solid white;">35.04022</td><td style="border: 1px solid white;">Albuquerque</td><td style="border: 1px solid white;">NM</td><td style="border: 1px solid white;">-106.60919</td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABQ</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">Aberdeen Regional Airport</td><td style="border: 1px solid white;">45.44906</td><td style="border: 1px solid white;">Aberdeen</td><td style="border: 1px solid white;">SD</td><td style="border: 1px solid white;">-98.42183</td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABR</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">Southwest Georgia Regional Airport</td><td style="border: 1px solid white;">31.53552</td><td style="border: 1px solid white;">Albany</td><td style="border: 1px solid white;">GA</td><td style="border: 1px solid white;">-84.19447</td><td style="border: 1px solid white;">USA</td><td style="border: 1px solid white;">ABY</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: airports, Number of rows: 322, Number of columns: 7
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
<p>And we also have access to the airlines names.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[107]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">airlines</span> <span class="o">=</span> <span class="n">vDataFrame</span><span class="p">(</span><span class="s2">&quot;airlines&quot;</span><span class="p">)</span>
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
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>AIRLINE</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>IATA_CODE</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">American Airlines Inc.</td><td style="border: 1px solid white;">AA</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">Alaska Airlines Inc.</td><td style="border: 1px solid white;">AS</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">JetBlue Airways</td><td style="border: 1px solid white;">B6</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">Delta Air Lines Inc.</td><td style="border: 1px solid white;">DL</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td><td style="border: 1px solid white;">EV</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
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
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can notice that each dataset has a primary or secondary key to join the data. For example we can join the 'flights' dataset to the 'airlines' and 'airport' datasets using the corresponding IATA code.</p>
<p>To join datasets in Vertica ML Python, use the 'join' method of the vDataFrame.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[108]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">help</span><span class="p">(</span><span class="n">vDataFrame</span><span class="o">.</span><span class="n">join</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>Help on function join in module vertica_ml_python.vdataframe:

join(self, input_relation, on:dict={}, how:str=&#39;natural&#39;, expr1:list=[&#39;*&#39;], expr2:list=[&#39;*&#39;])
    ---------------------------------------------------------------------------
    Joins the vDataFrame with another one or an input relation.
    
    Parameters
    ----------
    input_relation: str/vDataFrame
            Relation used to do the merging.
    on: dict, optional
            Dictionary of all the different keys. The dict must be similar to the following:
            {&#34;relationA_key1&#34;: &#34;relationB_key1&#34; ..., &#34;relationA_keyk&#34;: &#34;relationB_keyk&#34;}
            where relationA is the current vDataFrame and relationB is the input relation
            or the input vDataFrame.
    how: str, optional
            Join Type.
                    left    : Left Join.
                    right   : Right Join.
                    cross   : Cross Join.
                    full    : Full Outer Join.
                    natural : Natural Join.
                    inner   : Inner Join.
    expr1: list, optional
            List of the different columns to select from the current vDataFrame. 
            Pure SQL must be written. Aliases can also be given. &#39;column&#39; or 
            &#39;column AS my_new_alias&#39; are correct. Aliases are recommended to keep 
            the track of the different features and not have ambiguous names. 
    expr2: list, optional
            List of the different columns to select from the input relation. 
            Pure SQL must be written. Aliases can also be given. &#39;column&#39; or 
            &#39;column AS my_new_alias&#39; are correct. Aliases are recommended to keep 
            the track of the different features and not have ambiguous names. 
    
    Returns
    -------
    vDataFrame
            object result of the join.
    
    See Also
    --------
    vDataFrame.append  : Merges the vDataFrame with another relation.
    vDataFrame.groupby : Aggregates the vDataFrame.
    vDataFrame.sort    : Sorts the vDataFrame.

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
<p>Let's use a left join to merge the 'airlines' dataset and the 'flights' dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[121]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">flights</span> <span class="o">=</span> <span class="n">flights</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">airlines</span><span class="p">,</span>
                       <span class="n">how</span> <span class="o">=</span> <span class="s2">&quot;left&quot;</span><span class="p">,</span>
                       <span class="n">on</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;airline&quot;</span><span class="p">:</span> <span class="s2">&quot;IATA_CODE&quot;</span><span class="p">},</span>
                       <span class="n">expr2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;AIRLINE AS airline_long&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">flights</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline_long</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-01 10:09:00</td><td style="border: 1px solid white;">-9</td><td style="border: 1px solid white;">-2</td><td style="border: 1px solid white;">11433</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-01 10:27:00</td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">-14</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-01 13:57:00</td><td style="border: 1px solid white;">-4</td><td style="border: 1px solid white;">6</td><td style="border: 1px solid white;">13930</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-01 14:02:00</td><td style="border: 1px solid white;">-3</td><td style="border: 1px solid white;">-8</td><td style="border: 1px solid white;">11433</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">10135</td><td style="border: 1px solid white;">2015-10-01 14:44:00</td><td style="border: 1px solid white;">0</td><td style="border: 1px solid white;">-1</td><td style="border: 1px solid white;">10397</td><td style="border: 1px solid white;">EV</td><td style="border: 1px solid white;">Atlantic Southeast Airlines</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: join, Number of rows: 4068736, Number of columns: 7
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
<p>Let's use two left joins to get the information on the origin and destination airports.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[126]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">flights</span> <span class="o">=</span> <span class="n">flights</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">airports</span><span class="p">,</span>
                       <span class="n">how</span> <span class="o">=</span> <span class="s2">&quot;left&quot;</span><span class="p">,</span>
                       <span class="n">on</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;origin_airport&quot;</span><span class="p">:</span> <span class="s2">&quot;IATA_CODE&quot;</span><span class="p">},</span>
                       <span class="n">expr2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;LATITUDE AS origin_lat&quot;</span><span class="p">,</span>
                                <span class="s2">&quot;LONGITUDE AS origin_lon&quot;</span><span class="p">])</span>
<span class="n">flights</span> <span class="o">=</span> <span class="n">flights</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">airports</span><span class="p">,</span>
                       <span class="n">how</span> <span class="o">=</span> <span class="s2">&quot;left&quot;</span><span class="p">,</span>
                       <span class="n">on</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;destination_airport&quot;</span><span class="p">:</span> <span class="s2">&quot;IATA_CODE&quot;</span><span class="p">},</span>
                       <span class="n">expr2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;LATITUDE AS destination_lat&quot;</span><span class="p">,</span>
                                <span class="s2">&quot;LONGITUDE AS destination_lon&quot;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="n">flights</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>scheduled_departure</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>departure_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>arrival_delay</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_airport</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airline_long</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_lat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>origin_lon</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_lat</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>destination_lon</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">BRW</td><td style="border: 1px solid white;">2015-01-02 17:28:00</td><td style="border: 1px solid white;">-6</td><td style="border: 1px solid white;">-18</td><td style="border: 1px solid white;">ANC</td><td style="border: 1px solid white;">AS</td><td style="border: 1px solid white;">Alaska Airlines Inc.</td><td style="border: 1px solid white;">61.17432</td><td style="border: 1px solid white;">-149.99619</td><td style="border: 1px solid white;">71.28545</td><td style="border: 1px solid white;">-156.766</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">BRW</td><td style="border: 1px solid white;">2015-01-03 17:21:00</td><td style="border: 1px solid white;">-7</td><td style="border: 1px solid white;">-11</td><td style="border: 1px solid white;">SCC</td><td style="border: 1px solid white;">AS</td><td style="border: 1px solid white;">Alaska Airlines Inc.</td><td style="border: 1px solid white;">70.19476</td><td style="border: 1px solid white;">-148.46516</td><td style="border: 1px solid white;">71.28545</td><td style="border: 1px solid white;">-156.766</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">BRW</td><td style="border: 1px solid white;">2015-01-01 17:21:00</td><td style="border: 1px solid white;">-7</td><td style="border: 1px solid white;">-9</td><td style="border: 1px solid white;">SCC</td><td style="border: 1px solid white;">AS</td><td style="border: 1px solid white;">Alaska Airlines Inc.</td><td style="border: 1px solid white;">70.19476</td><td style="border: 1px solid white;">-148.46516</td><td style="border: 1px solid white;">71.28545</td><td style="border: 1px solid white;">-156.766</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">BRW</td><td style="border: 1px solid white;">2015-01-02 17:21:00</td><td style="border: 1px solid white;">-7</td><td style="border: 1px solid white;">2</td><td style="border: 1px solid white;">SCC</td><td style="border: 1px solid white;">AS</td><td style="border: 1px solid white;">Alaska Airlines Inc.</td><td style="border: 1px solid white;">70.19476</td><td style="border: 1px solid white;">-148.46516</td><td style="border: 1px solid white;">71.28545</td><td style="border: 1px solid white;">-156.766</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">BRW</td><td style="border: 1px solid white;">2015-01-03 17:28:00</td><td style="border: 1px solid white;">-7</td><td style="border: 1px solid white;">-20</td><td style="border: 1px solid white;">ANC</td><td style="border: 1px solid white;">AS</td><td style="border: 1px solid white;">Alaska Airlines Inc.</td><td style="border: 1px solid white;">61.17432</td><td style="border: 1px solid white;">-149.99619</td><td style="border: 1px solid white;">71.28545</td><td style="border: 1px solid white;">-156.766</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;object&gt;  Name: join, Number of rows: 4068736, Number of columns: 11
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
<p>Splitting the data into different tables is very important to avoid duplicated information and to save data storage. Just imagine writing the longitude and the latitude of the destination and origin airports for each flight. It will add too much duplicates and it can drastically increase the data volume.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Cross Joins are special. They don't need a key and they are used to do mathematical operations. Let's now use a cross join of the 'airports' dataset on itself to compute the distance between all the different airports.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[112]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">distances</span> <span class="o">=</span> <span class="n">airports</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">airports</span><span class="p">,</span> 
                          <span class="n">how</span> <span class="o">=</span> <span class="s2">&quot;cross&quot;</span><span class="p">,</span> 
                          <span class="n">expr1</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;IATA_CODE AS airport1&quot;</span><span class="p">,</span> 
                                   <span class="s2">&quot;LATITUDE AS airport1_latitude&quot;</span><span class="p">,</span> 
                                   <span class="s2">&quot;LONGITUDE AS airport1_longitude&quot;</span><span class="p">],</span>
                          <span class="n">expr2</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;IATA_CODE AS airport2&quot;</span><span class="p">,</span> 
                                   <span class="s2">&quot;LATITUDE AS airport2_latitude&quot;</span><span class="p">,</span> 
                                   <span class="s2">&quot;LONGITUDE AS airport2_longitude&quot;</span><span class="p">])</span>
<span class="n">distances</span><span class="o">.</span><span class="n">filter</span><span class="p">(</span><span class="s2">&quot;airport1 != airport2&quot;</span><span class="p">)</span>
<span class="n">distances</span><span class="o">.</span><span class="n">eval</span><span class="p">(</span><span class="s2">&quot;distance&quot;</span><span class="p">,</span> 
    <span class="s2">&quot;DISTANCE(airport1_latitude, airport1_longitude, airport2_latitude, airport2_longitude)&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>322 element(s) was/were filtered
</pre>
</div>
</div>

<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airport1</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airport1_latitude</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airport1_longitude</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airport2</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airport2_latitude</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>airport2_longitude</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>distance</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>0</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.65236</td><td style="border: 1px solid white;">-75.4404</td><td style="border: 1px solid white;">ABI</td><td style="border: 1px solid white;">32.41132</td><td style="border: 1px solid white;">-99.6819</td><td style="border: 1px solid white;">2341.90022515853</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.65236</td><td style="border: 1px solid white;">-75.4404</td><td style="border: 1px solid white;">ABQ</td><td style="border: 1px solid white;">35.04022</td><td style="border: 1px solid white;">-106.60919</td><td style="border: 1px solid white;">2791.44167745523</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.65236</td><td style="border: 1px solid white;">-75.4404</td><td style="border: 1px solid white;">ABR</td><td style="border: 1px solid white;">45.44906</td><td style="border: 1px solid white;">-98.42183</td><td style="border: 1px solid white;">1934.49820074978</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.65236</td><td style="border: 1px solid white;">-75.4404</td><td style="border: 1px solid white;">ABY</td><td style="border: 1px solid white;">31.53552</td><td style="border: 1px solid white;">-84.19447</td><td style="border: 1px solid white;">1281.62374218022</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>4</b></td><td style="border: 1px solid white;">ABE</td><td style="border: 1px solid white;">40.65236</td><td style="border: 1px solid white;">-75.4404</td><td style="border: 1px solid white;">ACK</td><td style="border: 1px solid white;">41.25305</td><td style="border: 1px solid white;">-70.06018</td><td style="border: 1px solid white;">456.66493443057</td></tr><tr><td style="border-top: 1px solid white;background-color:#214579;color:white"></td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td><td style="border: 1px solid white;">...</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[112]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;  Name: join, Number of rows: 103362, Number of columns: 7</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We now know how to merge different datasets together. In the next lesson, we will look at other interesting concepts like handling duplicates.</p>

</div>
</div>
</div>