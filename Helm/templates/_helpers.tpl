{{/*
Expand the name of the chart.
*/}}
{{- define "monai-inference.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "monai-inference.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s" .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "monai-inference.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Backend service name (used by nginx proxy_pass)
*/}}
{{- define "monai-inference.backendServiceName" -}}
{{- printf "%s-backend" (include "monai-inference.fullname" .) }}
{{- end }}

{{/*
Backend ServiceAccount name
*/}}
{{- define "monai-inference.backendServiceAccountName" -}}
{{- printf "%s-backend" (include "monai-inference.fullname" .) }}
{{- end }}

{{/*
Frontend ServiceAccount name
*/}}
{{- define "monai-inference.frontendServiceAccountName" -}}
{{- printf "%s-frontend" (include "monai-inference.fullname" .) }}
{{- end }}

{{/*
=============================================================================
NODE SELECTOR HELPERS
=============================================================================
nodeTarget controls physical node placement:
  "wn"   → backend on wn1 (node=worker1), frontend on wn2 (node=worker2)
  "c845" → both on c845 (node=worker3)

These labels exist on the nodes (verified via oc get no --show-labels):
  wn1:  node=worker1
  wn2:  node=worker2
  c845: node=worker3
=============================================================================
*/}}

{{/*
Node selector for the backend pod.
*/}}
{{- define "monai-inference.backendNodeSelector" -}}
{{- if eq .Values.nodeTarget "c845" }}
node: worker3
{{- else }}
node: worker1
{{- end }}
{{- end }}

{{/*
Node selector for the frontend pod.
*/}}
{{- define "monai-inference.frontendNodeSelector" -}}
{{- if eq .Values.nodeTarget "c845" }}
node: worker3
{{- else }}
node: worker2
{{- end }}
{{- end }}

{{/*
=============================================================================
GPU PRODUCT AFFINITY
=============================================================================
Adds a preferredDuringScheduling nodeAffinity rule that targets the correct
GPU product for the chosen nodeTarget.
  wn   → NVIDIA-L40S         (wn1/wn2, Ada Lovelace, 46 GB)
  c845 → NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition  (6× Blackwell, 97 GB)

This is a "preferred" rule — scheduling still succeeds if the product label
is absent, but will always win when the NVIDIA GPU Feature Discovery plugin
has labelled the node.
=============================================================================
*/}}
{{- define "monai-inference.gpuProductAffinity" -}}
{{- if .Values.backend.gpu.productAffinity }}
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
            - key: nvidia.com/gpu.product
              operator: In
              values:
                {{- if eq .Values.nodeTarget "c845" }}
                - NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition
                {{- else }}
                - NVIDIA-L40S
                {{- end }}
{{- end }}
{{- end }}
