#!/usr/bin/env python

#@HEADER
# ************************************************************************
#     Genten: Software for Generalized Tensor Decompositions
#     by Sandia National Laboratories
#
# Sandia National Laboratories is a multimission laboratory managed
# and operated by National Technology and Engineering Solutions of Sandia,
# LLC, a wholly owned subsidiary of Honeywell International, Inc., for the
# U.S. Department of Energy's National Nuclear Security Administration under
# contract DE-NA0003525.
#
# Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ************************************************************************
#@HEADER

import re
import os
from optparse import OptionParser

try:
    from matplotlib.pyplot import *
    has_mpl = True
except ImportError:
    has_mpl = False

def get_stats(file_name):
    f = open(file_name, 'r')
    its = 0
    cpals = 0
    mttkrp = 0
    fillComplete = 0
    for line in f:
        m = re.search('CpAls completed (\d+) iterations in (.*) seconds', line)
        if m != None:
            its = int(m.group(1))
            cpals = float(m.group(2))

        m = re.search('MTTKRP total time = (.*) seconds, average time = (.*) seconds', line)
        if m != None:
            mttkrp = float(m.group(2))

        m = re.search('Creating permutation arrays for perm MTTKRP method took \s*(.*) seconds', line)
        if m != None:
            fillComplete = float(m.group(1))

    return its,cpals,mttkrp,fillComplete,cpals+fillComplete

def run_cpals(options):
    output_filename = options.output + '_' + str(options.threads)
    path = os.path.join(os.curdir, output_filename)
    if not os.path.exists(path) or options.force:
        cmd = options.launch + ' ' + options.exe
        cmd = cmd + ' --kokkos-threads=' + str(options.threads)
        cmd = cmd + ' ' + options.args + ' >> ' + output_filename + ' 2>&1'
        out_file = open(output_filename, 'w')
        out_file.write(cmd + '\n')
        out_file.close()
        os.environ["OMP_NUM_THREADS"] = str(options.threads)
        os.environ["OMP_PROC_BIND"] = options.bind
        os.environ["OMP_PLACES"] = options.places
        os.system(cmd)
    return get_stats(output_filename)

def run_cpals_scaling(options, thread_range, affinity_range):
    its = []
    cpals = []
    mttkrp = []
    fillComplete = []
    total = []

    p = 2
    w = p + 6
    s = ''
    sep = '  '
    s = s + '{0:^{width}}'.format('n', width=3) + sep
    s = s + '{0:^{width}}'.format('its', width=3) + sep
    s = s + '{0:^{width}}'.format('cpals', width=w) + sep
    s = s + '{0:^{width}}'.format('mttkrp', width=w) + sep
    s = s + '{0:^{width}}'.format('fillCom', width=w) + sep
    print s
    s = '-'*3 + sep + '-'*3 + sep + '-'*w + sep + '-'*w + sep + '-'*w +sep
    print s

    n = len(thread_range)
    for i in range(n):
        threads = thread_range[i]
        affinity = affinity_range[i]
        options.threads = threads
        options.affinity = affinity
        i,c,m,f,t = run_cpals(options)
        its.append(i)
        cpals.append(c)
        mttkrp.append(m)
        fillComplete.append(f)
        total.append(t)
        s = ''
        s = s + '{0:>{width}d}'.format(threads, width=3) + sep
        s = s + '{0:>{width}d}'.format(i, width=3) + sep
        s = s + '{0:>{width}.{precision}e}'.format(c, width=w, precision=p) + sep
        s = s + '{0:>{width}.{precision}e}'.format(m, width=w, precision=p) + sep
        s = s + '{0:>{width}.{precision}e}'.format(f, width=w, precision=p) + sep
        print s
    return its,cpals,mttkrp,fillComplete,total

def parse_args(parser = OptionParser()):
    parser.add_option('--exe', default='./bin/perf_CpAlsRandomKtensor',
                      help='CpALS executable')
    parser.add_option('--launch', default='',
                      help='Command needed to launch executable')
    parser.add_option('-f', '--force', action="store_true",
                      dest="force", default=False,
                      help='Force run even if output file exists')
    parser.add_option('--threads', default=1,
                      help='Number of threads')
    parser.add_option('--numa', default=1,
                      help='Number of numa domains')
    parser.add_option('--bind', default='close',
                      help='Thread binding (OMP_PROC_BIND)')
    parser.add_option('--places', default='threads',
                      help='Thread placement (OMP_PLACES)')
    parser.add_option('--output', default='cpals.out',
                      help='Output filename')
    parser.add_option('--args', default='',
                      help='Command arguments')
    (options, args) = parser.parse_args()
    return options

#
# Script starts here
#

parser = OptionParser()
parser.add_option('--arch', default='',
                  help='Architecture to run on')
parser.add_option('--title', default='3K x 4K x 5K, 1M NNZ, R=32',
                  help='Plot title')
parser.add_option("-s", "--save", dest="save", default=None,
                  help="save figure to FILE", metavar="FILE")
options = parse_args(parser)

# Create thread ranges based on architcture
n = [ 1 ]
if options.arch == 'snb':
    n = [ 2**i for i in range(0,6) ]
if options.arch == 'hsw':
    n = [ 2**i for i in range(0,7) ]
elif options.arch == 'knl':
    n = [ 2**i for i in range(0,9) ]
elif options.arch == 'gpu' or options.arch == 'p100' or options.arch == 'k80':
    n = [ 1 ]

# Use compact affinity
a = [ 'compact' ]*len(n)

args = options.args
output = options.output

print '\nKokkos:'
options.args = args + ' --mttkrp-method atomic'
options.output = output + '_kokkos'
its_k,cpals_k,mttkrp_k,fillComplete_k,tot_k = run_cpals_scaling(options, n, a)

print '\nPerm:'
options.args = args + ' --mttkrp-method perm'
options.output = output + '_perm'
its_p,cpals_p,mttkrp_p,fillComplete_p,tot_p = run_cpals_scaling(options, n, a)

print '\nOrig Kokkos:'
options.args = args + ' --mttkrp-method orig-kokkos'
options.output = output + '_orig'
its_o,cpals_o,mttkrp_o,fillComplete_o,tot_o = run_cpals_scaling(options, n, a)

if not has_mpl:
    quit()

params = { 'font.size': 12,
           'font.weight': 'bold',
           'lines.linewidth': 2,
           'axes.linewidth': 2,
           'axes.fontweight': 'bold',
           'axes.labelweight': 'bold',
           'xtick.major.size': 8,
           'xtick.major.width': 2,
           'ytick.major.size': 8,
           'ytick.minor.size': 4
}
rcParams.update(params)

def ticksx(y, pos):
    return r'{0:.0f}'.format(y)
def ticksy(y, pos):
    return r'{0:.3f}'.format(y)

# Use bar chart if there is only one data point
if len(n) == 1:
    figure(1)
    clf()
    gca().set_yscale('log', basey=2)
    bar_width = 0.25
    ind_k = range(3)
    ind_p = [ i+bar_width for i in ind_k ]
    ind_t = [ i+bar_width/2 for i in ind_p ]
    kokkos = [ cpals_k[0], mttkrp_k[0], fillComplete_k[0] ]
    perm   = [ cpals_p[0], mttkrp_p[0], fillComplete_p[0] ]
    bar(ind_k,kokkos,bar_width,color='b',label='Kokkos')
    bar(ind_p,perm,bar_width,color='r',label='Perm')
    legend()
    ylabel('Time (s)')
    xticks(ind_t, ('CP-ALS (total)', 'MTTKRP (avg)', 'fillComplete'))
    title(options.arch.upper() + ', ' + options.title, fontsize=16)

    add_text = False
    if add_text:
        for i in range(len(kokkos)-1):
            text(ind_k[i]+bar_width/2, kokkos[i], str(kokkos[i]), horizontalalignment='center', verticalalignment='bottom', rotation=0)
        for i in range(len(perm)):
            text(ind_p[i]+bar_width/2, perm[i], str(perm[i]), horizontalalignment='center', verticalalignment='bottom', rotation=0)

else:
    figure(1, figsize=(8.5,11))
    clf()
    subplot(3,1,1)
    loglog(n,tot_k,'b-*',
           n,tot_p,'r-s',
           basex=2, basey=2)
    legend(('Kokkos','Perm'))
    xlabel('Threads')
    ylabel('Time (s)')
    title('CP-ALS (total)')
    gca().xaxis.set_major_formatter(FuncFormatter(ticksx))
    #gca().yaxis.set_major_formatter(FuncFormatter(ticksy))

    subplot(3,1,2)
    loglog(n,mttkrp_k,'b-*',
           n,mttkrp_p,'r-s',
           basex=2, basey=2)
    xlabel('Threads')
    ylabel('Time (s)')
    title('MTTKRP (average)')
    gca().xaxis.set_major_formatter(FuncFormatter(ticksx))
    #gca().yaxis.set_major_formatter(FuncFormatter(ticksy))

    subplot(3,1,3)
    loglog(n,fillComplete_k,'b-*',
           n,fillComplete_p,'r-s',
           basex=2, basey=2)
    xlabel('Threads')
    ylabel('Time (s)')
    title('fillComplete()')
    gca().xaxis.set_major_formatter(FuncFormatter(ticksx))
    #gca().yaxis.set_major_formatter(FuncFormatter(ticksy))

    subplots_adjust(hspace=0.5)
    suptitle(options.arch.upper() + ', ' + options.title, fontsize=16)

draw()
if options.save == None:
    show()

if options.save != None:
    savefig(options.save)
