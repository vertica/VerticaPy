<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="cross_validate">cross_validate<a class="anchor-link" href="#cross_validate">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cross_validate</span><span class="p">(</span><span class="n">estimator</span><span class="p">,</span> 
               <span class="n">input_relation</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
               <span class="n">X</span><span class="p">:</span> <span class="nb">list</span><span class="p">,</span> 
               <span class="n">y</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> 
               <span class="n">cv</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span> 
               <span class="n">pos_label</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span> 
               <span class="n">cutoff</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Computes the K-Fold cross validation of an estimator.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">estimator</div></td> <td><div class="type">object</div></td> <td><div class = "no">&#10060;</div></td> <td>Vertica estimator having a fit method and a DB cursor.</td> </tr>
    <tr> <td><div class="param_name">input_relation</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Input Relation.</td> </tr>
    <tr> <td><div class="param_name">X</div></td> <td><div class="type">list</div></td> <td><div class = "no">&#10060;</div></td> <td>List of the predictor columns.</td> </tr>
    <tr> <td><div class="param_name">y</div></td> <td><div class="type">str</div></td> <td><div class = "no">&#10060;</div></td> <td>Response Column.</td> </tr>
    <tr> <td><div class="param_name">cv</div></td> <td><div class="type">int</div></td> <td><div class = "yes">&#10003;</div></td> <td>Number of folds.</td> </tr>
    <tr> <td><div class="param_name">pos_label</div></td> <td><div class="type">int / float / str</div></td> <td><div class = "yes">&#10003;</div></td> <td>The main class to be considered as positive (classification only).</td> </tr>
    <tr> <td><div class="param_name">cutoff</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>The model cutoff (classification only).</td> </tr>
</table><h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><a href="../../utilities/tablesample/index.php">tablesample</a> : An object containing the result. For more information, check out <a href="../../utilities/tablesample/index.php">utilities.tablesample</a>.</p>

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
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">vertica_ml_python.learn.linear_model</span> <span class="k">import</span> <span class="n">LogisticRegression</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">LogisticRegression</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.LR_titanic&quot;</span><span class="p">,</span>
                           <span class="n">tol</span> <span class="o">=</span> <span class="mf">1e-4</span><span class="p">,</span> 
                           <span class="n">C</span> <span class="o">=</span> <span class="mf">1.0</span><span class="p">,</span> 
                           <span class="n">max_iter</span> <span class="o">=</span> <span class="mi">100</span><span class="p">,</span> 
                           <span class="n">solver</span> <span class="o">=</span> <span class="s1">&#39;CGD&#39;</span><span class="p">,</span>
                           <span class="n">l1_ratio</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">)</span>

<span class="kn">from</span> <span class="nn">vertica_ml_python.learn.model_selection</span> <span class="k">import</span> <span class="n">cross_validate</span>
<span class="n">cross_validate</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> 
               <span class="n">input_relation</span> <span class="o">=</span> <span class="s2">&quot;public.titanic_clean&quot;</span><span class="p">,</span> 
               <span class="n">X</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">,</span> <span class="s2">&quot;fare&quot;</span><span class="p">,</span> <span class="s2">&quot;parch&quot;</span><span class="p">,</span> <span class="s2">&quot;sex&quot;</span><span class="p">,</span> <span class="s2">&quot;boat&quot;</span><span class="p">],</span> 
               <span class="n">y</span> <span class="o">=</span> <span class="s2">&quot;survived&quot;</span><span class="p">,</span> 
               <span class="n">cv</span> <span class="o">=</span> <span class="mi">3</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt"></div>



<div class="output_html rendered_html output_subarea ">
<table style="border-collapse: collapse; border: 2px solid white"><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b></b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>auc</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>prc_auc</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>accuracy</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>log_loss</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>precision</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>recall</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>f1_score</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>mcc</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>informedness</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>markedness</b></td><td style="font-size:1.02em;background-color:#214579;color:white"><b>csi</b></td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>1-fold</b></td><td style="border: 1px solid white;">0.8351716885149669</td><td style="border: 1px solid white;">0.6998249233793346</td><td style="border: 1px solid white;">0.7695961995249406</td><td style="border: 1px solid white;">0.260097154064988</td><td style="border: 1px solid white;">0.9013157894736842</td><td style="border: 1px solid white;">0.6255707762557078</td><td style="border: 1px solid white;">0.7466157634750171</td><td style="border: 1px solid white;">0.5734536450307397</td><td style="border: 1px solid white;">0.5513133505131336</td><td style="border: 1px solid white;">0.5964830757190374</td><td style="border: 1px solid white;">0.5854700854700855</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>2-fold</b></td><td style="border: 1px solid white;">0.8178953322379943</td><td style="border: 1px solid white;">0.6477620824819167</td><td style="border: 1px solid white;">0.7622549019607843</td><td style="border: 1px solid white;">0.267716301128923</td><td style="border: 1px solid white;">0.8540145985401459</td><td style="border: 1px solid white;">0.6030927835051546</td><td style="border: 1px solid white;">0.7243194945272362</td><td style="border: 1px solid white;">0.5389170603312952</td><td style="border: 1px solid white;">0.5096348395799208</td><td style="border: 1px solid white;">0.5698817572117325</td><td style="border: 1px solid white;">0.5467289719626168</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>3-fold</b></td><td style="border: 1px solid white;">0.8569255936405695</td><td style="border: 1px solid white;">0.7364034561270908</td><td style="border: 1px solid white;">0.7970297029702971</td><td style="border: 1px solid white;">0.259948917187969</td><td style="border: 1px solid white;">0.9565217391304348</td><td style="border: 1px solid white;">0.6724890829694323</td><td style="border: 1px solid white;">0.7909265996148085</td><td style="border: 1px solid white;">0.6401381707835412</td><td style="border: 1px solid white;">0.6324890829694323</td><td style="border: 1px solid white;">0.6478797638217928</td><td style="border: 1px solid white;">0.652542372881356</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>avg</b></td><td style="border: 1px solid white;">0.8366642047978435</td><td style="border: 1px solid white;">0.6946634873294474</td><td style="border: 1px solid white;">0.7762936014853407</td><td style="border: 1px solid white;">0.26258745746062667</td><td style="border: 1px solid white;">0.9039507090480883</td><td style="border: 1px solid white;">0.6337175475767649</td><td style="border: 1px solid white;">0.7539539525390206</td><td style="border: 1px solid white;">0.5841696253818587</td><td style="border: 1px solid white;">0.5644790910208289</td><td style="border: 1px solid white;">0.6047481989175209</td><td style="border: 1px solid white;">0.5949138101046861</td></tr><tr style="{border: 1px solid white;}"><td style="border-bottom: 1px solid #DDD;font-size:1.02em;background-color:#214579;color:white"><b>std</b></td><td style="border: 1px solid white;">0.019557889199229246</td><td style="border: 1px solid white;">0.0445455227491905</td><td style="border: 1px solid white;">0.018329295978765273</td><td style="border: 1px solid white;">0.004442327269944722</td><td style="border: 1px solid white;">0.05130434259272791</td><td style="border: 1px solid white;">0.03540817712128269</td><td style="border: 1px solid white;">0.03390447427834591</td><td style="border: 1px solid white;">0.05145437276326108</td><td style="border: 1px solid white;">0.06247634610025867</td><td style="border: 1px solid white;">0.0396504281805422</td><td style="border: 1px solid white;">0.053535099745195935</td></tr></table>
</div>

</div>

<div class="output_area">

<div class="prompt output_prompt">Out[1]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>&lt;object&gt;</pre>
</div>

</div>

</div>
</div>

</div>