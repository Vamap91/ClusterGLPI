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


class GLPIClusteringSystem:
    def __init__(self, use_openai=True):
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.client = None
        self.kmeans = None
        self.clusters_info = {}
        self.embeddings = None
        
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
                
                if 'Urgência' in cluster_data.columns:
                    urgencia_dist = cluster_data['Urgência'].value_counts().to_dict()
                
                if 'Status' in cluster_data.columns:
                    status_dist = cluster_data['Status'].value_counts().to_dict()
                
                self.clusters_info[cluster_id] = {
                    'nome': cluster_name,
                    'total_chamados': len(cluster_data),
                    'sistema_principal': sistemas.index[0] if not sistemas.empty else 'Geral',
                    'keywords': keywords[:8],
                    'exemplos_titulos': titulos_originais.index.tolist()[:5],
                    'distribuicao_sistemas': sistemas.to_dict(),
                    'urgencia_distribuicao': urgencia_dist,
                    'status_distribuicao': status_dist,
                    'usado_ai': ai_name is not None
                }
    
    def predict_cluster(self, new_titles):
        """Prediz cluster para novos títulos"""
        if self.kmeans is None:
            raise ValueError("Modelo não foi treinado. Execute fit_clusters primeiro.")
        
        processed_titles = [self.preprocess_text(title) for title in new_titles]
        
        # Obter embeddings para novos títulos
        if self.use_openai and self.client:
            try:
                X_new = self.get_openai_embeddings(processed_titles)
            except:
                X_new = self.get_traditional_embeddings(processed_titles)
        else:
            X_new = self.get_traditional_embeddings(processed_titles)
        
        clusters = self.kmeans.predict(X_new)
        return clusters
    
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
                'IA_Usado': '🤖' if info['usado_ai'] else '📊'
            })
        
        return pd.DataFrame(summary_data)


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
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Download Excel com Clusters</a>'
    return href


def main():
    st.set_page_config(
        page_title="GLPI - Sistema de Clusterização de Chamados",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🎯 Sistema de Clusterização de Chamados GLPI")
    st.markdown("Sistema inteligente para agrupamento automático de chamados usando IA")
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
    else:
        st.sidebar.warning("⚠️ OpenAI não configurado")
        if not OPENAI_AVAILABLE:
            st.sidebar.info("Instale: pip install openai")
        else:
            st.sidebar.info("Adicione OPENAI_API_KEY nos secrets")
    
    # Sidebar para configurações
    st.sidebar.header("⚙️ Configurações")
    
    # Opção de usar OpenAI
    use_openai = st.sidebar.checkbox(
        "🤖 Usar OpenAI para embeddings", 
        value=openai_configured,
        disabled=not openai_configured,
        help="Melhora a precisão da clusterização usando embeddings da OpenAI"
    )
    
    # Upload do arquivo
    uploaded_file = st.file_uploader(
        "📁 Carregue o arquivo CSV/Excel dos chamados",
        type=['csv', 'xlsx', 'xls'],
        help="Arquivo deve conter uma coluna 'Título' com os títulos dos chamados"
    )
    
    if uploaded_file is not None:
        try:
            # Leitura do arquivo
            if uploaded_file.name.endswith('.csv'):
                # Tentar diferentes delimitadores
                try:
                    df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
                except:
                    df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Arquivo carregado com sucesso! {len(df)} registros encontrados.")
            
            # Mostrar preview dos dados
            with st.expander("👀 Preview dos Dados", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.info(f"Colunas encontradas: {', '.join(df.columns)}")
            
            # Verificação da coluna Título
            if 'Título' not in df.columns:
                st.error("❌ Coluna 'Título' não encontrada no arquivo.")
                st.write("**Colunas disponíveis:**", list(df.columns))
                st.stop()
            
            # Estatísticas básicas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total de Registros", len(df))
            with col2:
                titulos_validos = df['Título'].notna().sum()
                st.metric("📝 Títulos Válidos", titulos_validos)
            with col3:
                if 'Urgência' in df.columns:
                    alta_urgencia = (df['Urgência'] == 'Alta').sum()
                    st.metric("🔥 Alta Urgência", alta_urgencia)
            
            # Configurações do clustering
            st.sidebar.subheader("🔧 Parâmetros de Clustering")
            auto_clusters = st.sidebar.checkbox("Determinar clusters automaticamente", value=True)
            
            if not auto_clusters:
                n_clusters = st.sidebar.slider("Número de clusters", min_value=2, max_value=20, value=8)
            else:
                n_clusters = 8
            
            # Botão para executar clustering
            if st.sidebar.button("🚀 Executar Clusterização", type="primary"):
                try:
                    with st.spinner("Processando clusterização..."):
                        # Inicializa o sistema
                        clustering_system = GLPIClusteringSystem(use_openai=use_openai)
                        
                        # Executa clustering
                        df_clustered = clustering_system.fit_clusters(
                            df, 
                            auto_clusters=auto_clusters, 
                            n_clusters=n_clusters
                        )
                        
                        # Salva resultados na sessão
                        st.session_state['df_clustered'] = df_clustered
                        st.session_state['clustering_system'] = clustering_system
                        st.session_state['clusters_info'] = clustering_system.clusters_info
                        st.session_state['use_openai'] = use_openai
                    
                    st.success("✅ Clusterização concluída com sucesso!")
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erro durante a clusterização: {str(e)}")
                    st.exception(e)
            
            # Mostra resultados se disponíveis
            if 'df_clustered' in st.session_state:
                df_clustered = st.session_state['df_clustered']
                clustering_system = st.session_state['clustering_system']
                
                # Métricas principais
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total de Chamados", len(df_clustered))
                
                with col2:
                    st.metric("🎯 Clusters Identificados", len(clustering_system.clusters_info))
                
                with col3:
                    avg_per_cluster = len(df_clustered) / len(clustering_system.clusters_info)
                    st.metric("📈 Média por Cluster", f"{avg_per_cluster:.1f}")
                
                with col4:
                    ai_clusters = sum(1 for info in clustering_system.clusters_info.values() if info['usado_ai'])
                    st.metric("🤖 Clusters com IA", f"{ai_clusters}/{len(clustering_system.clusters_info)}")
                
                # Tabs para visualização
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📋 Resumo dos Clusters", 
                    "📊 Visualizações", 
                    "📝 Dados Detalhados", 
                    "🎯 Recomendações",
                    "📥 Download"
                ])
                
                with tab1:
                    st.subheader("📋 Resumo dos Clusters Identificados")
                    
                    cluster_summary = clustering_system.get_cluster_summary()
                    st.dataframe(cluster_summary, use_container_width=True)
                    
                    # Detalhes de cada cluster
                    st.subheader("🔍 Detalhes dos Clusters")
                    
                    for cluster_id, info in clustering_system.clusters_info.items():
                        with st.expander(f"Cluster {cluster_id}: {info['nome']} ({info['total_chamados']} chamados) {'🤖' if info['usado_ai'] else '📊'}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Palavras-chave:**")
                                st.write(", ".join(info['keywords']))
                                
                                st.write("**Sistema Principal:**")
                                st.write(info['sistema_principal'])
                                
                                if info['urgencia_distribuicao']:
                                    st.write("**Distribuição de Urgência:**")
                                    for urgencia, count in info['urgencia_distribuicao'].items():
                                        st.write(f"• {urgencia}: {count}")
                            
                            with col2:
                                st.write("**Exemplos de Títulos:**")
                                for titulo in info['exemplos_titulos']:
                                    st.write(f"• {titulo}")
                
                with tab2:
                    st.subheader("📊 Visualizações dos Clusters")
                    
                    # Gráfico de distribuição dos clusters
                    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
                    cluster_names = [f"C{i}: {clustering_system.clusters_info[i]['nome']}" for i in cluster_counts.index]
                    
                    fig_bar = px.bar(
                        x=cluster_names,
                        y=cluster_counts.values,
                        title="Distribuição de Chamados por Cluster",
                        labels={'x': 'Clusters', 'y': 'Número de Chamados'},
                        color=cluster_counts.values,
                        color_continuous_scale='viridis'
                    )
                    fig_bar.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Gráficos em duas colunas
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gráfico de pizza dos sistemas
                        sistema_counts = df_clustered['sistema_prefix'].value_counts()
                        fig_pie = px.pie(
                            values=sistema_counts.values,
                            names=sistema_counts.index,
                            title="Distribuição por Sistema"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Gráfico de urgência se disponível
                        if 'Urgência' in df_clustered.columns:
                            urgencia_counts = df_clustered['Urgência'].value_counts()
                            fig_urgencia = px.bar(
                                x=urgencia_counts.index,
                                y=urgencia_counts.values,
                                title="Distribuição por Urgência",
                                color=urgencia_counts.values,
                                color_continuous_scale='reds'
                            )
                            st.plotly_chart(fig_urgencia, use_container_width=True)
                
                with tab3:
                    st.subheader("📝 Dados Detalhados com Clusters")
                    
                    # Filtros
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        cluster_filter = st.multiselect(
                            "Filtrar por Cluster:",
                            options=sorted(df_clustered['cluster'].unique()),
                            default=sorted(df_clustered['cluster'].unique())
                        )
                    
                    with col2:
                        sistema_filter = st.multiselect(
                            "Filtrar por Sistema:",
                            options=sorted(df_clustered['sistema_prefix'].unique()),
                            default=sorted(df_clustered['sistema_prefix'].unique())
                        )
                    
                    with col3:
                        if 'Urgência' in df_clustered.columns:
                            urgencia_filter = st.multiselect(
                                "Filtrar por Urgência:",
                                options=sorted(df_clustered['Urgência'].unique()),
                                default=sorted(df_clustered['Urgência'].unique())
                            )
                        else:
                            urgencia_filter = []
                    
                    # Aplicar filtros
                    df_filtered = df_clustered[df_clustered['cluster'].isin(cluster_filter)]
                    df_filtered = df_filtered[df_filtered['sistema_prefix'].isin(sistema_filter)]
                    
                    if urgencia_filter and 'Urgência' in df_clustered.columns:
                        df_filtered = df_filtered[df_filtered['Urgência'].isin(urgencia_filter)]
                    
                    # Adicionar nome do cluster
                    df_filtered['Nome_Cluster'] = df_filtered['cluster'].map(
                        lambda x: clustering_system.clusters_info[x]['nome']
                    )
                    
                    # Reorganizar colunas
                    cols_order = ['ID', 'Título', 'cluster', 'Nome_Cluster', 'sistema_prefix']
                    if 'Urgência' in df_filtered.columns:
                        cols_order.append('Urgência')
                    if 'Status' in df_filtered.columns:
                        cols_order.append('Status')
                    
                    # Adicionar outras colunas
                    cols_order += [col for col in df_filtered.columns 
                                  if col not in cols_order + ['titulo_processado']]
                    
                    df_display = df_filtered[cols_order]
                    
                    st.dataframe(df_display, use_container_width=True)
                    st.info(f"📊 Mostrando {len(df_filtered)} de {len(df_clustered)} registros")
                
                with tab4:
                    st.subheader("🎯 Recomendações para Direcionamento")
                    
                    # Análise de especialização por desenvolvedor
                    if 'Atribuído - Técnico' in df_clustered.columns:
                        st.write("**Sugestões de Especialização por Desenvolvedor:**")
                        
                        tecnico_cluster = df_clustered.groupby(['Atribuído - Técnico', 'cluster']).size().reset_index(name='count')
                        
                        for cluster_id, info in clustering_system.clusters_info.items():
                            cluster_tecnicos = tecnico_cluster[tecnico_cluster['cluster'] == cluster_id]
                            if not cluster_tecnicos.empty:
                                top_tecnico = cluster_tecnicos.loc[cluster_tecnicos['count'].idxmax()]
                                
                                st.write(f"**{info['nome']}** → {top_tecnico['Atribuído - Técnico']} ({top_tecnico['count']} chamados)")
                    
                    # Análise de carga de trabalho
                    st.write("**Análise de Distribuição de Carga:**")
                    for cluster_id, info in clustering_system.clusters_info.items():
                        if info['total_chamados'] > len(df_clustered) * 0.2:  # Mais de 20% dos chamados
                            st.warning(f"⚠️ Cluster '{info['nome']}' tem alta concentração ({info['total_chamados']} chamados)")
                        elif info['total_chamados'] < 5:
                            st.info(f"ℹ️ Cluster '{info['nome']}' tem poucos chamados ({info['total_chamados']})")
                
                with tab5:
                    st.subheader("📥 Download dos Resultados")
                    
                    # Preparar dados para download
                    df_download = df_clustered.copy()
                    df_download['Nome_Cluster'] = df_download['cluster'].map(
                        lambda x: clustering_system.clusters_info[x]['nome']
                    )
                    
                    # Remover colunas técnicas
                    columns_to_remove = ['titulo_processado']
                    df_download = df_download.drop(columns=[col for col in columns_to_remove if col in df_download.columns])
                    
                    # Reorganizar colunas
                    cols_final = ['ID', 'Título', 'cluster', 'Nome_Cluster', 'sistema_prefix']
                    if 'Urgência' in df_download.columns:
                        cols_final.append('Urgência')
                    if 'Status' in df_download.columns:
                        cols_final.append('Status')
                    
                    # Adicionar outras colunas
                    cols_final += [col for col in df_download.columns if col not in cols_final]
                    
                    df_download = df_download[cols_final]
                    
                    # Estatísticas do download
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Total de Registros", len(df_download))
                    with col2:
                        st.metric("🎯 Clusters", df_download['cluster'].nunique())
                    with col3:
                        ai_used = st.session_state.get('use_openai', False)
                        st.metric("🤖 IA Utilizada", "Sim" if ai_used else "Não")
                    
                    # Link para download
                    download_link = create_download_link(df_download, "chamados_clusterizados.xlsx")
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                    st.success("✅ Arquivo pronto para download com os clusters aplicados!")
                    
                    # Preview dos dados
                    st.subheader("👀 Preview dos Dados para Download")
                    st.dataframe(df_download.head(10), use_container_width=True)
                    
                    # Informações sobre o arquivo
                    st.info("""
                    **O arquivo Excel contém:**
                    - Aba 'Chamados_Clusterizados': Todos os chamados com clusters aplicados
                    - Aba 'Resumo_Clusters': Resumo detalhado de cada cluster
                    - Coluna 'cluster': ID numérico do cluster
                    - Coluna 'Nome_Cluster': Nome descritivo do cluster
                    """)
        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
            st.exception(e)
    
    else:
        # Página inicial quando não há arquivo
        st.info("📁 Carregue um arquivo CSV ou Excel para começar a clusterização.")
        
        # Instruções e benefícios
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Formato do Arquivo")
            st.write("""
            **Coluna obrigatória:**
            - **Título**: Títulos dos chamados para clusterização
            
            **Colunas opcionais que serão preservadas:**
            - ID
            - Urgência
            - Status
            - Data de abertura
            - Atribuído - Técnico
            - Requerente
            - Outras colunas do seu sistema
            
            **Formatos aceitos:**
            - CSV (separado por ; ou ,)
            - Excel (.xlsx, .xls)
            """)
        
        with col2:
            st.subheader("🚀 Benefícios do Sistema")
            st.write("""
            **🎯 Clusterização Inteligente:**
            - Agrupa chamados similares automaticamente
            - Identifica padrões nos títulos
            - Facilita direcionamento para especialistas
            
            **🤖 Powered by AI:**
            - Usa embeddings da OpenAI para melhor precisão
            - Nomes de clusters gerados por IA
            - Fallback para método tradicional
            
            **📊 Análises Completas:**
            - Visualizações interativas
            - Recomendações de direcionamento
            - Estatísticas detalhadas
            """)
        
        # Exemplo de dados
        st.subheader("💡 Exemplo de Dados")
        example_data = {
            'ID': ['3347', '15862', '18612'],
            'Título': [
                'SQ | Tela de log espelho do WA',
                'WA | Otimização - Acompanhamento de Peça',
                'WA | Atendimento presencial'
            ],
            'Urgência': ['Média', 'Alta', 'Média'],
            'Status': ['Em atendimento', 'Em atendimento', 'Em atendimento']
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🎯 Sistema de Clusterização GLPI | Desenvolvido com ❤️ usando Streamlit e OpenAI</p>
    </div>
    """, unsafe_allow_html=True)


# Configuração para deployment no Streamlit Cloud
if __name__ == "__main__":
    # Configurações específicas para Streamlit Cloud
    try:
        main()
    except Exception as e:
        st.error(f"Erro na aplicação: {str(e)}")
        st.info("Verifique se todas as dependências estão instaladas e a chave da OpenAI está configurada nos secrets.")
