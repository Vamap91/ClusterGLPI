import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import time
import json

# Importação condicional da OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("⚠️ OpenAI não instalado. Usando apenas método tradicional.")


class CriticalityAnalyzer:
    """Analisador de criticidade baseado na Matriz de Impacto x Urgência"""
    
    def __init__(self):
        # Palavras-chave para IMPACTO (baseado na matriz fornecida)
        self.impacto_keywords = {
            1: [  # Muito Alto - Áreas críticas
                'financeiro', 'faturamento', 'boleto', 'pagamento', 'reembolso',
                'nota fiscal', 'receita', 'cobrança', 'api atendimento', 
                'cliente', 'produção', 'estoque', 'pedido', 'integração'
            ],
            2: [  # Alto - Parte significativa do negócio
                'vendas', 'comercial', 'parceiro', 'integração', 'sistema vendas',
                'crm', 'erp', 'relatório gerencial'
            ],
            3: [  # Médio - Área não crítica mas recorrente
                'administrativo', 'relatório', 'rh', 'interno', 'gestão',
                'dashboard', 'controle'
            ],
            4: [  # Baixo - Impacto limitado
                'exibição', 'layout', 'visual', 'interface', 'tela',
                'formulário', 'campo'
            ],
            5: [  # Muito Baixo - Melhorias
                'melhoria', 'otimização', 'performance', 'estética',
                'sugestão', 'funcionalidade nova'
            ]
        }
        
        # Palavras-chave para URGÊNCIA
        self.urgencia_keywords = {
            1: [  # Muito Urgente - Imediato
                'parado', 'bloqueado', 'fora do ar', 'não funciona',
                'erro crítico', 'sistema indisponível', 'urgente',
                'imediato', 'crítico'
            ],
            2: [  # Urgente - Hoje
                'problema', 'falha', 'erro', 'bug', 'não carrega',
                'lento', 'travado', 'timeout'
            ],
            3: [  # Moderado - 48h
                'dificuldade', 'demora', 'inconsistência',
                'retrabalho', 'manual'
            ],
            4: [  # Baixo - Semana
                'ajuste', 'configuração', 'parametrização',
                'pequeno erro', 'correção'
            ],
            5: [  # Planejável
                'agendamento', 'futuro', 'próxima versão',
                'quando possível', 'melhoria'
            ]
        }
        
        # Matriz de Prioridade (Impacto x Urgência)
        self.priority_matrix = {
            (1, 1): 'P1', (1, 2): 'P2', (1, 3): 'P3', (1, 4): 'P4', (1, 5): 'P5',
            (2, 1): 'P2', (2, 2): 'P3', (2, 3): 'P4', (2, 4): 'P5', (2, 5): 'Planejado',
            (3, 1): 'P3', (3, 2): 'P4', (3, 3): 'P5', (3, 4): 'Planejado', (3, 5): 'Backlog',
            (4, 1): 'P4', (4, 2): 'P5', (4, 3): 'Planejado', (4, 4): 'Planejado', (4, 5): 'Backlog',
            (5, 1): 'P5', (5, 2): 'Planejado', (5, 3): 'Planejado', (5, 4): 'Planejado', (5, 5): 'Backlog'
        }
        
        # SLAs por prioridade (em horas)
        self.sla_hours = {
            'P1': 2, 'P2': 4, 'P3': 8, 'P4': 24, 'P5': 48,
            'Planejado': 168, 'Backlog': 720  # 1 semana / 1 mês
        }
    
    def detect_impacto(self, titulo):
        """Detecta o impacto baseado em palavras-chave"""
        if pd.isna(titulo):
            return 5
        
        titulo_lower = str(titulo).lower()
        
        for impacto_level, keywords in self.impacto_keywords.items():
            for keyword in keywords:
                if keyword in titulo_lower:
                    return impacto_level
        
        return 3  # Médio como padrão
    
    def detect_urgencia(self, titulo):
        """Detecta a urgência baseada em palavras-chave"""
        if pd.isna(titulo):
            return 5
        
        titulo_lower = str(titulo).lower()
        
        for urgencia_level, keywords in self.urgencia_keywords.items():
            for keyword in keywords:
                if keyword in titulo_lower:
                    return urgencia_level
        
        return 3  # Moderado como padrão
    
    def calculate_priority(self, impacto, urgencia):
        """Calcula a prioridade baseada na matriz"""
        return self.priority_matrix.get((impacto, urgencia), 'P5')
    
    def get_sla_hours(self, prioridade):
        """Retorna SLA em horas para a prioridade"""
        return self.sla_hours.get(prioridade, 48)
    
    def get_criticality_words(self, titulo):
        """Retorna palavras-chave de criticidade encontradas"""
        if pd.isna(titulo):
            return ''
        
        titulo_lower = str(titulo).lower()
        found_words = []
        
        # Busca em todas as categorias
        all_keywords = []
        for keywords_dict in [self.impacto_keywords, self.urgencia_keywords]:
            for keywords_list in keywords_dict.values():
                all_keywords.extend(keywords_list)
        
        for keyword in all_keywords:
            if keyword in titulo_lower:
                found_words.append(keyword)
        
        return ', '.join(found_words)
    
    def analyze_dataframe(self, df, titulo_col='Título'):
        """Analisa DataFrame completo e adiciona colunas de criticidade"""
        df = df.copy()
        
        # Aplicar análise
        df['Impacto_Detectado'] = df[titulo_col].apply(self.detect_impacto)
        df['Urgencia_Detectada'] = df[titulo_col].apply(self.detect_urgencia)
        df['Prioridade'] = df.apply(lambda row: self.calculate_priority(
            row['Impacto_Detectado'], row['Urgencia_Detectada']), axis=1)
        df['SLA_Horas'] = df['Prioridade'].apply(self.get_sla_hours)
        df['Criticidade_Palavras'] = df[titulo_col].apply(self.get_criticality_words)
        
        return df


class GLPIClusteringSystem:
    def __init__(self, use_openai=True):
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.client = None
        self.kmeans = None
        self.clusters_info = {}
        self.embeddings = None
        self.criticality_analyzer = CriticalityAnalyzer()
        
        # Configurar OpenAI se disponível
        if self.use_openai:
            try:
                # Verificar se a chave existe nos secrets
                if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                    api_key = st.secrets['OPENAI_API_KEY']
                    if api_key and api_key.startswith('sk-'):
                        self.client = OpenAI(api_key=api_key)
                        st.success("✅ OpenAI configurado com sucesso!")
                    else:
                        st.warning("⚠️ Chave da OpenAI inválida. Usando método tradicional.")
                        self.use_openai = False
                else:
                    st.info("ℹ️ Chave da OpenAI não encontrada. Usando método tradicional.")
                    self.use_openai = False
            except Exception as e:
                st.warning(f"⚠️ Erro ao configurar OpenAI: {str(e)}. Usando método tradicional.")
                self.use_openai = False
                self.client = None
        
    def preprocess_text(self, text):
        """Pré-processamento específico para títulos de chamados técnicos"""
        if pd.isna(text) or text == '':
            return ''
        
        # Converter para minúsculas mantendo estrutura técnica
        text = str(text).lower()
        
        # Manter separadores importantes (|, -, /)
        text = re.sub(r'[^\w\s\|\-\/\.]', ' ', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_system_prefix(self, title):
        """Extrai o prefixo do sistema (WA, SQ, Sistema, etc.)"""
        if pd.isna(title):
            return 'Outros'
        
        title_str = str(title).strip()
        
        # Padrões comuns identificados nos dados
        if title_str.startswith('WA |'):
            return 'WA'
        elif title_str.startswith('SQ |'):
            return 'SQ'
        elif title_str.startswith('Sistema -'):
            # Extrai o subsistema
            parts = title_str.split(' - ')
            if len(parts) > 1:
                subsystem = parts[1].split(' ')[0]
                return f"Sistema - {subsystem}"
            return 'Sistema'
        else:
            return 'Outros'
    
    def get_openai_embeddings(self, texts, batch_size=50):
        """Obtém embeddings usando OpenAI API com rate limiting"""
        if not self.client:
            raise ValueError("Cliente OpenAI não configurado")
        
        embeddings = []
        
        # Processar em lotes menores para evitar rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                with st.spinner(f"Obtendo embeddings {i+1}-{min(i+batch_size, len(texts))} de {len(texts)}..."):
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch,
                        encoding_format="float"
                    )
                    
                    batch_embeddings = [data.embedding for data in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    # Rate limiting - pausa entre lotes
                    if i + batch_size < len(texts):
                        time.sleep(0.5)
                        
            except Exception as e:
                st.error(f"Erro ao obter embeddings: {str(e)}")
                st.info("Tentando com método tradicional...")
                raise e
        
        return np.array(embeddings)
    
    def get_traditional_embeddings(self, texts):
        """Método tradicional usando TF-IDF como fallback"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X = vectorizer.fit_transform(texts)
        return X.toarray()
    
    def determine_optimal_clusters(self, X, max_clusters=15):
        """Determina o número ótimo de clusters usando método do cotovelo e silhouette"""
        silhouette_scores = []
        inertias = []
        k_range = range(2, min(max_clusters + 1, len(X)))
        
        with st.spinner("Determinando número ótimo de clusters..."):
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                inertias.append(kmeans.inertia_)
        
        # Escolhe o k com melhor silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        return best_k, silhouette_scores, inertias
    
    def generate_cluster_name_with_ai(self, cluster_titles, cluster_id):
        """Gera nome do cluster usando OpenAI"""
        if not self.client:
            return None
        
        try:
            # Prepara os títulos mais representativos
            sample_titles = cluster_titles[:8]  # Reduzido para evitar tokens em excesso
            titles_text = "\n".join([f"- {title}" for title in sample_titles])
            
            prompt = f"""Analise os títulos de chamados técnicos e crie um nome descritivo:

{titles_text}

Crie um nome conciso (máximo 4 palavras) que represente o tema principal.
Responda apenas com o nome, sem explicações."""
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um especialista em análise de chamados técnicos."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=30,
                temperature=0.1
            )
            
            cluster_name = response.choices[0].message.content.strip()
            return cluster_name if cluster_name else None
            
        except Exception as e:
            st.warning(f"Erro ao gerar nome com AI para cluster {cluster_id}: {str(e)}")
            return None
    
    def fit_clusters(self, df, titulo_col='Título', auto_clusters=True, n_clusters=8):
        """Ajusta o modelo de clusterização aos dados"""
        # Pré-processamento
        df = df.copy()
        df['titulo_processado'] = df[titulo_col].apply(self.preprocess_text)
        df['sistema_prefix'] = df[titulo_col].apply(self.extract_system_prefix)
        
        # NOVA FUNCIONALIDADE: Análise de Criticidade
        with st.spinner("Analisando criticidade dos chamados..."):
            df = self.criticality_analyzer.analyze_dataframe(df, titulo_col)
        
        # Remove títulos vazios
        df_clean = df[df['titulo_processado'] != ''].copy()
        
        if len(df_clean) == 0:
            raise ValueError("Nenhum título válido encontrado para clusterização")
        
        # Obter embeddings
        texts = df_clean['titulo_processado'].tolist()
        
        if self.use_openai and self.client:
            try:
                # Usar embeddings da OpenAI
                st.info("🤖 Usando embeddings da OpenAI para melhor precisão...")
                X = self.get_openai_embeddings(texts)
                st.success("✅ Embeddings da OpenAI obtidos com sucesso!")
            except Exception as e:
                st.warning("⚠️ Erro com OpenAI, usando método tradicional...")
                X = self.get_traditional_embeddings(texts)
        else:
            # Usar método tradicional
            st.info("📊 Usando método tradicional TF-IDF...")
            X = self.get_traditional_embeddings(texts)
        
        # Salvar embeddings
        self.embeddings = X
        
        # Determinação do número de clusters
        if auto_clusters:
            optimal_k, silhouette_scores, inertias = self.determine_optimal_clusters(X)
            n_clusters = optimal_k
            st.info(f"🎯 Número ótimo de clusters determinado: {n_clusters}")
        
        # Aplicação do K-Means
        with st.spinner("Executando algoritmo de clusterização..."):
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = self.kmeans.fit_predict(X)
        
        # Adiciona clusters ao DataFrame
        df_clean['cluster'] = clusters
        
        # Análise dos clusters
        self.analyze_clusters(df_clean)
        
        return df_clean
    
    def analyze_clusters(self, df):
        """Analisa e nomeia os clusters baseado no conteúdo"""
        self.clusters_info = {}
        
        with st.spinner("Analisando e nomeando clusters..."):
            for cluster_id in sorted(df['cluster'].unique()):
                cluster_data = df[df['cluster'] == cluster_id]
                
                # Títulos mais comuns no cluster
                titulos_originais = cluster_data['Título'].value_counts().head(10)
                
                # Sistemas mais comuns
                sistemas = cluster_data['sistema_prefix'].value_counts()
                
                # Palavras-chave mais importantes
                titulos_cluster = cluster_data['titulo_processado'].tolist()
                all_words = ' '.join(titulos_cluster).split()
                word_freq = Counter(all_words)
                keywords = [word for word, freq in word_freq.most_common(15) 
                           if len(word) > 2 and word not in ['sistema', 'tela', 'para', 'com', 'por', 'dos', 'das']]
                
                # Tentar gerar nome com AI primeiro
                ai_name = None
                if self.use_openai and self.client:
                    ai_name = self.generate_cluster_name_with_ai(titulos_originais.index.tolist(), cluster_id)
                
                # Nome do cluster
                if ai_name:
                    cluster_name = ai_name
                else:
                    # Fallback para método tradicional
                    sistema_principal = sistemas.index[0] if not sistemas.empty else 'Geral'
                    keyword_principal = keywords[0] if keywords else 'Diversos'
                    cluster_name = f"{sistema_principal} - {keyword_principal.title()}"
                
                # Análise de urgência e status se disponível
                urgencia_dist = {}
                status_dist = {}
                prioridade_dist = {}
                
                if 'Urgência' in cluster_data.columns:
                    urgencia_dist = cluster_data['Urgência'].value_counts().to_dict()
                
                if 'Status' in cluster_data.columns:
                    status_dist = cluster_data['Status'].value_counts().to_dict()
                
                # NOVA: Distribuição de prioridades
                prioridade_dist = cluster_data['Prioridade'].value_counts().to_dict()
                
                # NOVA: Análise de criticidade
                criticidade_media = {
                    'impacto_medio': cluster_data['Impacto_Detectado'].mean(),
                    'urgencia_media': cluster_data['Urgencia_Detectada'].mean(),
                    'sla_medio': cluster_data['SLA_Horas'].mean()
                }
                
                self.clusters_info[cluster_id] = {
                    'nome': cluster_name,
                    'total_chamados': len(cluster_data),
                    'sistema_principal': sistemas.index[0] if not sistemas.empty else 'Geral',
                    'keywords': keywords[:8],
                    'exemplos_titulos': titulos_originais.index.tolist()[:5],
                    'distribuicao_sistemas': sistemas.to_dict(),
                    'urgencia_distribuicao': urgencia_dist,
                    'status_distribuicao': status_dist,
                    'prioridade_distribuicao': prioridade_dist,
                    'criticidade_media': criticidade_media,
                    'usado_ai': ai_name is not None
                }
    
    def get_cluster_summary(self):
        """Retorna resumo dos clusters"""
        if not self.clusters_info:
            return pd.DataFrame()
        
        summary_data = []
        for cluster_id, info in self.clusters_info.items():
            summary_data.append({
                'Cluster_ID': cluster_id,
                'Nome_Cluster': info['nome'],
                'Total_Chamados': info['total_chamados'],
                'Sistema_Principal': info['sistema_principal'],
                'Palavras_Chave': ', '.join(info['keywords'][:5]),
                'Exemplo_Titulo': info['exemplos_titulos'][0] if info['exemplos_titulos'] else '',
                'Impacto_Medio': round(info['criticidade_media']['impacto_medio'], 1),
                'Urgencia_Media': round(info['criticidade_media']['urgencia_media'], 1),
                'SLA_Medio_Horas': round(info['criticidade_media']['sla_medio'], 1),
                'IA_Usado': '🤖' if info['usado_ai'] else '📊'
            })
        
        return pd.DataFrame(summary_data)


def create_priority_matrix_heatmap(df):
    """Cria heatmap interativo da Matriz de Impacto x Urgência"""
    # Criar matriz de contagem
    matrix_data = df.groupby(['Impacto_Detectado', 'Urgencia_Detectada']).size().reset_index(name='count')
    
    # Criar matriz 5x5 completa
    matrix = np.zeros((5, 5))
    priority_labels = np.empty((5, 5), dtype=object)
    
    # Matriz de prioridades para labels
    priority_matrix = {
        (1, 1): 'P1', (1, 2): 'P2', (1, 3): 'P3', (1, 4): 'P4', (1, 5): 'P5',
        (2, 1): 'P2', (2, 2): 'P3', (2, 3): 'P4', (2, 4): 'P5', (2, 5): 'Plan',
        (3, 1): 'P3', (3, 2): 'P4', (3, 3): 'P5', (3, 4): 'Plan', (3, 5): 'Back',
        (4, 1): 'P4', (4, 2): 'P5', (4, 3): 'Plan', (4, 4): 'Plan', (4, 5): 'Back',
        (5, 1): 'P5', (5, 2): 'Plan', (5, 3): 'Plan', (5, 4): 'Plan', (5, 5): 'Back'
    }
    
    # Preencher matriz
    for _, row in matrix_data.iterrows():
        i = row['Impacto_Detectado'] - 1  # Converter para índice 0-4
        j = row['Urgencia_Detectada'] - 1
        matrix[i][j] = row['count']
    
    # Preencher labels
    for i in range(5):
        for j in range(5):
            priority = priority_matrix.get((i+1, j+1), '')
            count = int(matrix[i][j])
            priority_labels[i][j] = f"{priority}<br>{count}" if count > 0 else priority
    
    # Criar heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        text=priority_labels,
        texttemplate="%{text}",
        textfont={"size": 12},
        x=['Urgência 1<br>(Imediato)', 'Urgência 2<br>(Hoje)', 'Urgência 3<br>(48h)', 
           'Urgência 4<br>(Semana)', 'Urgência 5<br>(Planejável)'],
        y=['Impacto 5<br>(Muito Baixo)', 'Impacto 4<br>(Baixo)', 'Impacto 3<br>(Médio)', 
           'Impacto 2<br>(Alto)', 'Impacto 1<br>(Muito Alto)'],
        colorscale=[
            [0, '#f0f0f0'],      # Cinza claro para zero
            [0.2, '#4CAF50'],    # Verde para P5/Planejado
            [0.4, '#FFC107'],    # Amarelo para P4/P3
            [0.6, '#FF9800'],    # Laranja para P2
            [1.0, '#F44336']     # Vermelho para P1
        ],
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Chamados: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="🔥 Matriz de Impacto x Urgência - Chamados por Prioridade",
        xaxis_title="Urgência →",
        yaxis_title="Impacto ↑",
        width=800,
        height=500
    )
    
    return fig


def create_priority_distribution_chart(df):
    """Cria gráfico de distribuição de prioridades"""
    priority_counts = df['Prioridade'].value_counts()
    
    # Cores por prioridade
    colors = {
        'P1': '#F44336',      # Vermelho
        'P2': '#FF9800',      # Laranja
        'P3': '#FFC107',      # Amarelo
        'P4': '#4CAF50',      # Verde
        'P5': '#2196F3',      # Azul
        'Planejado': '#9E9E9E',  # Cinza
        'Backlog': '#607D8B'     # Cinza escuro
    }
    
    fig = px.bar(
        x=priority_counts.index,
        y=priority_counts.values,
        title="📊 Distribuição de Chamados por Prioridade",
        labels={'x': 'Prioridade', 'y': 'Número de Chamados'},
        color=priority_counts.index,
        color_discrete_map=colors
    )
    
    fig.update_layout(showlegend=False)
    return fig


def create_sla_analysis_chart(df):
    """Cria gráfico de análise de SLA"""
    sla_counts = df['SLA_Horas'].value_counts().sort_index()
    
    # Mapear horas para labels legíveis
    sla_labels = {
        2: '2h (P1)', 4: '4h (P2)', 8: '8h (P3)', 
        24: '1d (P4)', 48: '2d (P5)', 
        168: '1sem (Plan)', 720: '1mês (Back)'
    }
    
    labels = [sla_labels.get(hours, f'{hours}h') for hours in sla_counts.index]
    
    fig = px.pie(
        values=sla_counts.values,
        names=labels,
        title="⏱️ Distribuição de SLAs",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    return fig


def create_download_link(df, filename):
    """Cria link para download do DataFrame como Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Chamados_Clusterizados', index=False)
        
        # Adicionar uma aba com resumo dos clusters se disponível
        if 'clustering_system' in st.session_state:
            cluster_summary = st.session_state['clustering_system'].get_cluster_summary()
            cluster_summary.to_excel(writer, sheet_name='Resumo_Clusters', index=False)
    
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Download Excel com Clusters e Matriz</a>'
    return href


def main():
    st.set_page_config(
        page_title="GLPI - Sistema de Clusterização com Matriz de Prioridade",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🎯 Sistema de Clusterização GLPI + Matriz de Prioridade")
    st.markdown("Sistema inteligente para agrupamento e priorização automática de chamados usando IA")
    st.markdown("---")
    
    # Verificar configuração da OpenAI
    openai_configured = False
    if OPENAI_AVAILABLE and hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        api_key = st.secrets.get('OPENAI_API_KEY', '')
        if api_key and api_key.startswith('sk-'):
            openai_configured = True
            st.sidebar.success("🤖 OpenAI configurado")
        else:
            st.sidebar.warning("⚠️ Chave OpenAI inválida")
    else
