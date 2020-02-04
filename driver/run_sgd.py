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

def get_stats(file_name):
    f = open(file_name, 'r')
    tot = 0
    for line in f:
        m = re.search('GCP-SGD completed (\d+) iterations in (.*) seconds', line)
        if m != None:
            tot = float(m.group(2))

    return tot

def run_sgd(options):
    path = os.path.join(os.curdir, options.output)
    if not os.path.exists(path) or options.force:
        cmd = options.launch + ' ' + options.exe
        input_filename = options.input
        if options.gz:
            input_filename = input_filename + '.gz'
        cmd = cmd + ' --method gcp-sgd'
        cmd = cmd + ' --input ' + input_filename
        cmd = cmd + ' --index-base ' + str(options.index)
        if options.gz:
            cmd = cmd + ' --gz'
        cmd = cmd + ' --timings'
        cmd = cmd + ' --kokkos-threads=' + str(options.threads)
        cmd = cmd + ' ' + options.args + ' >> ' + options.output + ' 2>&1'
        out_file = open(options.output, 'w')
        out_file.write(cmd + '\n')
        out_file.close()
        os.environ["OMP_NUM_THREADS"] = str(options.threads)
        os.environ["OMP_PROC_BIND"] = options.bind
        os.environ["OMP_PLACES"] = options.places
        os.system(cmd)
    return get_stats(options.output)

def run_sgd_tensors(options, tensors):
    total = []

    p = 2
    w = p + 6
    s = ''
    sep = '  '
    s = s + '{0:^{width}}'.format('tensor', width=10) + sep
    print s
    #s = '-'*10 + sep + '-'*3 + sep + '-'*w + sep + '-'*w + sep + '-'*w +sep
    #print s

    output = options.output
    for tensor in tensors:
        options.input = tensor[0]
        options.output = output + '_' + tensor[1]
        options.output = options.output + '_' + str(options.threads)
        options.index = tensor[2]
        t = run_sgd(options)
        total.append(t)
        s = ''
        s = s + '{0:>{width}}'.format(tensor[1], width=10) + sep
        s = s + '{0:>{width}.{precision}e}'.format(t, width=w, precision=p) + sep
        print s
    return total

def parse_args(parser = OptionParser()):
    parser.add_option('--exe', default='./bin/genten',
                      help='Genten executable')
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
    parser.add_option('--input', default='sptensor.dat',
                      help='Input filename')
    parser.add_option('--output', default='sgd.out',
                      help='Output filename')
    parser.add_option('--args', default='',
                      help='Command arguments')
    parser.add_option('--gz', action="store_true", default=False,
                      help='Read compressed tensor')
    parser.add_option('--index', default=1,
                      help='Index base')
    parser.add_option('--cuda', action="store_true", default=False,
                      help='Whether we are running on cuda')
    (options, args) = parser.parse_args()
    return options

#
# Script starts here
#

parser = OptionParser()
options = parse_args(parser)

args = options.args
output = options.output

tensors = [ ('lbnl-network.tns', 'LBNL', 1),
            ('chicago-crime-comm.tns', 'Chicago', 1),
            ('uber.tns', 'Uber', 1),
            ('vast-2015-mc1-5d.tns', 'VAST', 1),
            ('enron.tns', 'Enron', 1),
            ('nell-2.tns', 'NELL2', 1) ]
#            ('delicious-4d.tns', 'Delicious', 1),
#            ('delicious-3d.tns', 'Delicious3', 1) ]


#print '\nStratified Sort:'
#options.args = args + ' --sampling stratified';
#options.output = output + '_sort'
#tot_sort = run_sgd_tensors(options, tensors)

#print '\nStratified Hash:'
#options.args = args + '  --sampling stratified --hash'
#options.output = output + '_hash'
#tot_hash = run_sgd_tensors(options, tensors)

print '\nSemi-Stratified:'
options.args = args + '  --sampling semi-stratified --hash'
if options.cuda:
  options.args = options.args + ' --mttkrp-all-method atomic'
else:
  options.args = options.args + ' --mttkrp-all-method iterated --mttkrp-method perm'
options.output = output + '_semi'
tot_semi = run_sgd_tensors(options, tensors)

print '\nSemi-Stratified Fused Atomic:'
options.args = args + '  --sampling semi-stratified --hash --fuse --mttkrp-all-method atomic'
options.output = output + '_fuse'
tot_fuse_atomic = run_sgd_tensors(options, tensors)

print '\nSemi-Stratified Fused-SA:'
options.args = args + '  --sampling semi-stratified --hash --fuse-sa'
options.output = output + '_fuse_sa'
tot_fuse_sa = run_sgd_tensors(options, tensors)
