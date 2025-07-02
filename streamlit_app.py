import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import re
from collections import Counter
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64


class GLPIClusteringSystem:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Mantemos palavras técnicas
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.kmeans = None
        self.clusters_info = {}
        self.embeddings = None
        
    def preprocess_text(self, text):
        """Pré-processamento específico para títulos de chamados técnicos"""
        if pd.isna(text) or text == '':
            return ''
        
        # Converter para minúsculas
        text = str(text).lower()
        
        # Manter separadores importantes (|, -, /)
        text = re.sub(r'[^\w\s\|\-\/]', ' ', text)
        
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_system_prefix(self, title):
        """Extrai o prefixo do sistema (WA, SQ, Sistema, etc.)"""
        if pd.isna(title):
            return 'Outros'
        
        title_str = str(title).strip()
        
        # Padrões comuns identificados
        if title_str.startswith('WA |'):
            return 'WA'
        elif title_str.startswith('SQ |'):
            return 'SQ'
        elif title_str.startswith('Sistema -'):
            # Extrai o subsistema
            parts = title_str.split(' - ')
            if len(parts) > 1:
                return f"Sistema - {parts[1].split(' ')[0]}"
            return 'Sistema'
        else:
            return 'Outros'
    
    def determine_optimal_clusters(self, X, max_clusters=15):
        """Determina o número ótimo de clusters usando método do cotovelo e silhouette"""
        silhouette_scores = []
        inertias = []
        k_range = range(2, min(max_clusters + 1, len(X)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        # Escolhe o k com melhor silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        return best_k, silhouette_scores, inertias
    
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
        
        # Vetorização TF-IDF
        X = self.vectorizer.fit_transform(df_clean['titulo_processado'])
        
        # Determinação do número de clusters
        if auto_clusters:
            optimal_k, silhouette_scores, inertias = self.determine_optimal_clusters(X)
            n_clusters = optimal_k
        
        # Aplicação do K-Means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X)
        
        # Adiciona clusters ao DataFrame
        df_clean['cluster'] = clusters
        
        # Análise dos clusters
        self.analyze_clusters(df_clean)
        
        # Salva embeddings para uso futuro
        self.embeddings = X.toarray()
        
        return df_clean
    
    def analyze_clusters(self, df):
        """Analisa e nomeia os clusters baseado no conteúdo"""
        self.clusters_info = {}
        
        for cluster_id in df['cluster'].unique():
            cluster_data = df[df['cluster'] == cluster_id]
            
            # Títulos mais comuns no cluster
            titulos_originais = cluster_data['Título'].value_counts().head(5)
            
            # Sistemas mais comuns
            sistemas = cluster_data['sistema_prefix'].value_counts()
            
            # Palavras-chave mais importantes
            titulos_cluster = cluster_data['titulo_processado'].tolist()
            all_words = ' '.join(titulos_cluster).split()
            word_freq = Counter(all_words)
            keywords = [word for word, freq in word_freq.most_common(10) 
                       if len(word) > 2 and word not in ['sistema', 'tela', 'para']]
            
            # Gera nome do cluster
            sistema_principal = sistemas.index[0] if not sistemas.empty else 'Geral'
            keyword_principal = keywords[0] if keywords else 'Diversos'
            
            cluster_name = f"{sistema_principal} - {keyword_principal.title()}"
            
            self.clusters_info[cluster_id] = {
                'nome': cluster_name,
                'total_chamados': len(cluster_data),
                'sistema_principal': sistema_principal,
                'keywords': keywords[:5],
                'exemplos_titulos': titulos_originais.index.tolist(),
                'distribuicao_sistemas': sistemas.to_dict()
            }
    
    def predict_cluster(self, new_titles):
        """Prediz cluster para novos títulos"""
        if self.kmeans is None:
            raise ValueError("Modelo não foi treinado. Execute fit_clusters primeiro.")
        
        processed_titles = [self.preprocess_text(title) for title in new_titles]
        X_new = self.vectorizer.transform(processed_titles)
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
                'Palavras_Chave': ', '.join(info['keywords']),
                'Exemplo_Titulo': info['exemplos_titulos'][0] if info['exemplos_titulos'] else ''
            })
        
        return pd.DataFrame(summary_data)


def create_download_link(df, filename):
    """Cria link para download do DataFrame como Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Chamados_Clusterizados', index=False)
    
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 Download Excel com Clusters</a>'
    return href


def main():
    st.set_page_config(
        page_title="GLPI - Sistema de Clusterização de Chamados",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Sistema de Clusterização de Chamados GLPI")
    st.markdown("---")
    
    # Sidebar para configurações
    st.sidebar.header("⚙️ Configurações")
    
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
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Arquivo carregado com sucesso! {len(df)} registros encontrados.")
            
            # Verificação da coluna Título
            if 'Título' not in df.columns:
                st.error("❌ Coluna 'Título' não encontrada no arquivo. Verifique o formato.")
                st.stop()
            
            # Configurações do clustering
            st.sidebar.subheader("🔧 Parâmetros de Clustering")
            auto_clusters = st.sidebar.checkbox("Determinar clusters automaticamente", value=True)
            
            if not auto_clusters:
                n_clusters = st.sidebar.slider("Número de clusters", min_value=2, max_value=20, value=8)
            else:
                n_clusters = 8
            
            # Botão para executar clustering
            if st.sidebar.button("🚀 Executar Clusterização", type="primary"):
                with st.spinner("Processando clusterização..."):
                    # Inicializa o sistema
                    clustering_system = GLPIClusteringSystem()
                    
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
                
                st.success("✅ Clusterização concluída com sucesso!")
                st.experimental_rerun()
            
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
                    sistemas_unicos = df_clustered['sistema_prefix'].nunique()
                    st.metric("🔧 Sistemas Únicos", sistemas_unicos)
                
                # Tabs para visualização
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Resumo dos Clusters", "📊 Visualizações", "📝 Dados Detalhados", "📥 Download"])
                
                with tab1:
                    st.subheader("📋 Resumo dos Clusters Identificados")
                    
                    cluster_summary = clustering_system.get_cluster_summary()
                    st.dataframe(cluster_summary, use_container_width=True)
                    
                    # Detalhes de cada cluster
                    st.subheader("🔍 Detalhes dos Clusters")
                    
                    for cluster_id, info in clustering_system.clusters_info.items():
                        with st.expander(f"Cluster {cluster_id}: {info['nome']} ({info['total_chamados']} chamados)"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Palavras-chave:**")
                                st.write(", ".join(info['keywords']))
                                
                                st.write("**Sistema Principal:**")
                                st.write(info['sistema_principal'])
                            
                            with col2:
                                st.write("**Exemplos de Títulos:**")
                                for titulo in info['exemplos_titulos'][:3]:
                                    st.write(f"• {titulo}")
                
                with tab2:
                    st.subheader("📊 Visualizações dos Clusters")
                    
                    # Gráfico de distribuição dos clusters
                    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
                    cluster_names = [clustering_system.clusters_info[i]['nome'] for i in cluster_counts.index]
                    
                    fig_bar = px.bar(
                        x=cluster_names,
                        y=cluster_counts.values,
                        title="Distribuição de Chamados por Cluster",
                        labels={'x': 'Clusters', 'y': 'Número de Chamados'}
                    )
                    fig_bar.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Gráfico de pizza dos sistemas
                    sistema_counts = df_clustered['sistema_prefix'].value_counts()
                    fig_pie = px.pie(
                        values=sistema_counts.values,
                        names=sistema_counts.index,
                        title="Distribuição por Sistema"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with tab3:
                    st.subheader("📝 Dados Detalhados com Clusters")
                    
                    # Filtros
                    col1, col2 = st.columns(2)
                    
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
                    
                    # Aplicar filtros
                    df_filtered = df_clustered[
                        (df_clustered['cluster'].isin(cluster_filter)) &
                        (df_clustered['sistema_prefix'].isin(sistema_filter))
                    ]
                    
                    # Adicionar nome do cluster
                    df_filtered['Nome_Cluster'] = df_filtered['cluster'].map(
                        lambda x: clustering_system.clusters_info[x]['nome']
                    )
                    
                    # Reorganizar colunas
                    cols_order = ['ID', 'Título', 'cluster', 'Nome_Cluster', 'sistema_prefix'] + \
                                [col for col in df_filtered.columns if col not in ['ID', 'Título', 'cluster', 'Nome_Cluster', 'sistema_prefix', 'titulo_processado']]
                    
                    df_display = df_filtered[cols_order]
                    
                    st.dataframe(df_display, use_container_width=True)
                    
                    st.info(f"📊 Mostrando {len(df_filtered)} de {len(df_clustered)} registros")
                
                with tab4:
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
                    cols_final = ['ID', 'Título', 'cluster', 'Nome_Cluster', 'sistema_prefix'] + \
                                [col for col in df_download.columns if col not in ['ID', 'Título', 'cluster', 'Nome_Cluster', 'sistema_prefix']]
                    
                    df_download = df_download[cols_final]
                    
                    # Link para download
                    download_link = create_download_link(df_download, "chamados_clusterizados.xlsx")
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                    st.success("✅ Arquivo pronto para download com os clusters aplicados!")
                    
                    # Preview dos dados
                    st.subheader("👀 Preview dos Dados para Download")
                    st.dataframe(df_download.head(10), use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Erro ao processar arquivo: {str(e)}")
    
    else:
        st.info("📁 Carregue um arquivo CSV ou Excel para começar a clusterização.")
        
        # Informações sobre o formato esperado
        st.subheader("📋 Formato do Arquivo")
        st.write("""
        O arquivo deve conter pelo menos uma coluna chamada **'Título'** com os títulos dos chamados.
        
        **Formatos aceitos:**
        - CSV (separado por ponto e vírgula)
        - Excel (.xlsx, .xls)
        
        **Colunas opcionais que serão preservadas:**
        - ID
        - Urgência
        - Status
        - Data de abertura
        - Atribuído - Técnico
        - Requerente
        - Outras colunas do seu sistema
        """)


if __name__ == "__main__":
    main()
