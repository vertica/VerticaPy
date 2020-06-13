<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="classifier.score">classifier.score<a class="anchor-link" href="#classifier.score">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><b>Definition for Multiclass Classifier:</b></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">classifier</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">pos_label</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span> 
                 <span class="n">cutoff</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> 
                 <span class="n">method</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;accuracy&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><b>Definition for Binary Classifier:</b></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">classifier</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">cutoff</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">0.5</span><span class="p">,</span> 
                 <span class="n">method</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="s2">&quot;accuracy&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Computes the model score.</p>
<h3 id="Parameters">Parameters<a class="anchor-link" href="#Parameters">&#182;</a></h3><table id="parameters">
    <tr> <th>Name</th> <th>Type</th> <th>Optional</th> <th>Description</th> </tr>
    <tr> <td><div class="param_name">pos_label</div></td> <td><div class="type">int / float / str </div></td> <td><div class = "yes">&#10003;</div></td> <td>Label to consider as positive. All the other classes will be merged and considered as negative in case of multi classification.</td> </tr>
    <tr> <td><div class="param_name">cutoff</div></td> <td><div class="type">float</div></td> <td><div class = "yes">&#10003;</div></td> <td>Cutoff for which the tested category will be accepted as prediction. If the parameter is not between 0 and 1, an automatic cutoff is computed.</td> </tr>
    <tr> <td><div class="param_name">method</div></td> <td><div class="type">str</div></td> <td><div class = "yes">&#10003;</div></td> <td>The method used to compute the score.<br>
                                                    <ul>
                                                        <li><b>accuracy :</b> Accuracy</li>
                                                        <li><b>auc :</b> Area Under the Curve (ROC)</li>
                                                        <li><b>best_cutoff :</b> Cutoff which optimised the ROC Curve prediction.</li>
                                                        <li><b>bm :</b> Informedness = tpr + tnr - 1</li>
                                                        <li><b>csi :</b> Critical Success Index = tp / (tp + fn + fp)</li>
                                                        <li><b>f1 :</b> F1 Score</li>
                                                        <li><b>logloss :</b> Log Loss </li>
                                                        <li><b>mcc :</b> Matthews Correlation Coefficient</li>
                                                        <li><b>mk :</b> Markedness = ppv + npv - 1</li>
                                                        <li><b>npv :</b> Negative Predictive Value = tn / (tn + fn)</li>
                                                        <li><b>prc_auc :</b> Area Under the Curve (PRC)</li>
                                                        <li><b>precision :</b> Precision = tp / (tp + fp)</li>
                                                        <li><b>specificity :</b> Specificity = tn / (tn + fp)</li></ul></td> </tr>
</table>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div>
<div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Returns">Returns<a class="anchor-link" href="#Returns">&#182;</a></h3><p><b>float</b> : score</p>
<h3 id="Example">Example<a class="anchor-link" href="#Example">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Multiclass Classification</span>
<span class="kn">from</span> <span class="nn">vertica_ml_python.learn.ensemble</span> <span class="k">import</span> <span class="n">RandomForestClassifier</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.RF_iris&quot;</span><span class="p">,</span>
                               <span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span>
                               <span class="n">max_features</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">,</span>
                               <span class="n">max_leaf_nodes</span> <span class="o">=</span> <span class="mi">32</span><span class="p">,</span> 
                               <span class="n">sample</span> <span class="o">=</span> <span class="mf">0.7</span><span class="p">,</span>
                               <span class="n">max_depth</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span>
                               <span class="n">min_samples_leaf</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                               <span class="n">min_info_gain</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">,</span>
                               <span class="n">nbins</span> <span class="o">=</span> <span class="mi">32</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.iris&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;PetalLengthCm&quot;</span><span class="p">,</span> <span class="s2">&quot;PetalWidthCm&quot;</span><span class="p">],</span> <span class="s2">&quot;Species&quot;</span><span class="p">)</span>
<span class="c1"># Logloss</span>
<span class="n">model</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;log_loss&quot;</span><span class="p">,</span> <span class="n">pos_label</span> <span class="o">=</span> <span class="s2">&quot;Iris-virginica&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[26]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0.037429015840143</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Global accuracy</span>
<span class="n">model</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;accuracy&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[27]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0.946666666666667</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1"># Binary Classification</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">name</span> <span class="o">=</span> <span class="s2">&quot;public.RF_titanic&quot;</span><span class="p">,</span>
                               <span class="n">n_estimators</span> <span class="o">=</span> <span class="mi">20</span><span class="p">,</span>
                               <span class="n">max_features</span> <span class="o">=</span> <span class="s2">&quot;auto&quot;</span><span class="p">,</span>
                               <span class="n">max_leaf_nodes</span> <span class="o">=</span> <span class="mi">32</span><span class="p">,</span> 
                               <span class="n">sample</span> <span class="o">=</span> <span class="mf">0.7</span><span class="p">,</span>
                               <span class="n">max_depth</span> <span class="o">=</span> <span class="mi">3</span><span class="p">,</span>
                               <span class="n">min_samples_leaf</span> <span class="o">=</span> <span class="mi">5</span><span class="p">,</span>
                               <span class="n">min_info_gain</span> <span class="o">=</span> <span class="mf">0.0</span><span class="p">,</span>
                               <span class="n">nbins</span> <span class="o">=</span> <span class="mi">32</span><span class="p">)</span>
<span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="s2">&quot;public.titanic&quot;</span><span class="p">,</span> <span class="p">[</span><span class="s2">&quot;age&quot;</span><span class="p">,</span> <span class="s2">&quot;fare&quot;</span><span class="p">,</span> <span class="s2">&quot;sex&quot;</span><span class="p">],</span> <span class="s2">&quot;survived&quot;</span><span class="p">)</span>
<span class="c1"># AUC</span>
<span class="n">model</span><span class="o">.</span><span class="n">score</span><span class="p">(</span><span class="n">method</span> <span class="o">=</span> <span class="s2">&quot;auc&quot;</span><span class="p">)</span>
</pre></div>

</div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

<div class="prompt output_prompt">Out[28]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0.8348079727758869</pre>
</div>

</div>

</div>
</div>

</div>