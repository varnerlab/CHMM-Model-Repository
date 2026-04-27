Please fix these instructions. I noticed that for the discrete NJ and WJ HMM model, you used K, lambda, and epsilon with numbers that do not make any sense (too few states). Please refer to the paper and choose the best combination of parameters to run this model: https://arxiv.org/pdf/2603.10202

I noticed in some figures you omitted the labels for the axes, like in Figures 4 and 5. 

For Figure 4, the two panels are not aligned. Also, the captions can be improved. In general, the captions for tables and figures can be improved. 

In Section 4.5, you kept a reference to the v6 paper OoS window; please remove it. Also, check for similar nuances. 

For the random seed which we used to maintain reproducibility, I think you can mention it once in the body of the paper. 

In Section 4.6, signature-MMD, the equation at the end of "We lift each window to a 2D path (t/W, rt/std(r))" goes outside the paper margins. 

Similarly, for the leverage effect, "The observed leverage effect on the IS window, measured as avg 20k=1 corr(r^2_t r_{t−k})" goes outside the margin of the paper. 

Some table captions have statements like "reproduced from ......."; I think this is not necessary.

In the Section 4.3 title, it says "seven-model comparison." I counted 8. Maybe I am wrong, but please double-check that. 

In Table 3, it shows a comparison to 9 generators, so again, which one is correct?

I think it will be really great if we add either one or two schematics explaining the architecture of pipelines A and B, as well as the architectures of the models we present in this study. It can be one or more figures as you see fit. 

Generally, figures quality and choice of colors can be improved.

Figure 5 panel b, you can adjust the contrast between the color to make the simulations more apparent. Also there is no legend for the simulations. Check other figures for similar nuances. 

Also, check for similar nuances to these across the paper and update teh paper plan and decision memo files accordingly. 